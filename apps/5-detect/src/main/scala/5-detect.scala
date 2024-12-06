import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{StructType, StructField, StringType}
import com.linkedin.relevance.isolationforest._
import java.text.SimpleDateFormat
import java.util.Date
import scala.sys.process._

object ScalaApp {
  def main(args: Array[String]): Unit = {
  
    val spark = SparkSession.builder.appName("5-detect")
      .master("spark://spark-master:7077")
      .config("spark.hadoop.fs.s3a.endpoint", sys.env("S3_HOST"))
      .config("spark.hadoop.fs.s3a.access.key", sys.env("S3_USER"))
      .config("spark.hadoop.fs.s3a.secret.key", sys.env("S3_PASS"))
      .config("spark.hadoop.fs.s3a.path.style.access", "true")
      .getOrCreate()
  
    val start_time = new SimpleDateFormat("yy-MM-dd-HH_mm").format(new Date())

    val curr_domain = "canvas.ieec.mx"

    val df = spark.read
      .parquet("s3a://logs/output/4-domain_ground_truth/")
      .filter(col("domain_category") === curr_domain)
      .select(
        "method",
        "status",
        "body_bytes_sent",
        "http_referer",
        "user_agent",
        "dec_req_uri",
        "clean_path",
        "clean_query_list",
        "domain",
        "fdate_time",
        "fabstime",
        "day_of_week",
        "domain_category"
      )

    val df_time = df.withColumn("es_time", date_format(col("fdate_time"), "yyyy-MM-dd'T'HH:mm:ss:SSSZ"))

    val df_url_features = df_time
      .withColumn("url_features", 
        concat(
          split(col("clean_path"), "/"),
          transform(col("clean_query_list"), x => split(x, "=")(0))
        )
      )
      .withColumn("query_features",
        transform(col("clean_query_list"), x => split(x, "=")(1))
      )

    //###### Use Word2Vec
    //https://spark.apache.org/docs/latest/ml-features#word2vec
    val url_hashingTF = new HashingTF()
      .setInputCol("url_features")
      .setOutputCol("url_tf_features")
      .setNumFeatures(16384)
    
    val df_url_hash = url_hashingTF.transform(df_url_features)

    val query_hashingTF = new HashingTF()
      .setInputCol("query_features")
      .setOutputCol("query_tf_features")
      .setNumFeatures(8192)
    
    val df_hash = query_hashingTF.transform(df_url_hash)

    val vector_assembler = new VectorAssembler()
      .setInputCols(Array(
        "fabstime", 
        "status", 
        "body_bytes_sent", 
        "url_tf_features", 
        "query_tf_features"
      ))
      .setOutputCol("features")

    val df_vectorized = vector_assembler.transform(df_hash)

    val isolationForest = new IsolationForest()
      .setNumEstimators(100)
      .setBootstrap(false)
      .setMaxSamples(256)
      .setMaxFeatures(1.0)
      .setFeaturesCol("features")
      .setPredictionCol("anomaly")
      .setScoreCol("outlierScore")
      .setContamination(0.05)
      .setRandomSeed(1)

    val isolationForestModel = isolationForest.fit(df_vectorized)

    isolationForestModel.write.overwrite.save("s3a://logs/output/5-detect-model/" + curr_domain + "/")

    /*
    val isolationForestModel2 = IsolationForestModel.load()
    */

    val df_scores = isolationForestModel.transform(df_vectorized)
    
    // df_scores.filter(col("anomaly") === 1).show(truncate=false)
    // df_scores.filter(col("anomaly") === 1).count
    
    df_scores.write
      .partitionBy("domain_category")
      .parquet("s3a://logs/output/5-detect-predict/" + curr_domain + "/")


    val log = Seq(
      Row("Start", start_time),
      Row("End", new SimpleDateFormat("yy-MM-dd-HH_mm").format(new Date())),
      Row("domain_category", curr_domain),
      Row("Volume", s"${df.count()}, ${df.columns.length}"),
      Row("Anomalies", s"${df_scores.filter(col("anomaly") === 1).count}")
    )
    
    // Schema
    val log_schema = StructType(Seq(
      StructField("Metric", StringType, true),
      StructField("Value", StringType, true)
    ))

    val df_log = spark.createDataFrame(spark.sparkContext.parallelize(log), log_schema)

    df_log.coalesce(1).write
      .option("header", "true")
      .csv("s3a://logs/output/log/5-detect")

  
    val df2 = df_scores.withColumn("anomaly_bool", col("anomaly").cast(BooleanType))

    df2.select("fabstime", "status", "body_bytes_sent", "outlierScore", "anomaly_bool").write.format("es").mode("overwrite").save("anom")


  }
}
