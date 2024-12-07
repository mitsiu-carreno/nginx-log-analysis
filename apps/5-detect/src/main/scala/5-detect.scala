import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{NGram, VectorAssembler, HashingTF, MinHashLSH}
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

    var curr_domain: String = ""
   
    if (args.length > 0) {
      curr_domain = args(0) 
    } else {
      println()
      println("-" * 200)
      println("No domain provided for path:")
      println("s3a://logs/output/4-domain_ground_truth/domain_category=${curr_domain}/")
      println("-" * 200)

      // Stop the Spark session
      spark.stop()
    }

 
    val start_time = new SimpleDateFormat("yy-MM-dd-HH_mm").format(new Date())

    val df = spark.read.parquet("s3a://logs/output/4-domain_ground_truth/")
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

    val df_time = df.withColumn(
      "es_time", 
      date_format(col("fdate_time"), "yyyy-MM-dd'T'HH:mm:ss.SSSZ")
    )


    /** MINHASH Fail

    val df_chars = df_time.withColumn("url_chars", split(col("clean_path"), ""))

    val nGram = new NGram().setN(9).setInputCol("url_chars").setOutputCol("ngrams")

    val df_ngram = nGram.transform(df_chars)

    val hashingTF = new HashingTF().setInputCol("ngrams").setOutputCol("raw_hashes").setNumFeatures(16384)

    val df_raw_hash = hashingTF.transform(df_ngram) 

    val minHashLSH = new MinHashLSH().setInputCol("raw_hashes").setOutputCol("min_hashes").setNumHashTables(5)

    val df_minhash = minHashLSH.fit(df_raw_hash).transform(df_raw_hash)

    val df_test = df_minhash.select(col("method"),
      col("status"),
      col("body_bytes_sent"),
      col("http_referer"),
      col("user_agent"),
      col("dec_req_uri"),
      col("clean_path"),
      col("clean_query_list"),
      col("domain"),
      col("fdate_time"),
      col("fabstime"),
      col("day_of_week"),
      col("es_time"),
      col("min_hashes")(0).alias("min0"),
      col("min_hashes")(1).alias("min1"),
      col("min_hashes")(2).alias("min2"),
      col("min_hashes")(3).alias("min3"),
      col("min_hashes")(4).alias("min4")
    )

    val vector_assembler = new VectorAssembler()
      .setInputCols(Array(
        "fabstime",
        "status",
        "body_bytes_sent",
        "min0",
        "min1",
        "min2",
        "min3",
        "min4"
      ))
      .setOutputCol("features")
    
    val df_vectorized = vector_assembler.transform(df_test)

    **/ 

    val df_prep = df_time
      .withColumn(
        "query_len", 
        length(col("dec_req_uri")) - length(col("clean_path"))
      )
      .withColumn(
        "agent", 
        split(col("user_agent"), " ")
      )
      .withColumn(
        "url_features", 
        concat(
          split(col("clean_path"), "/"), 
          transform(col("clean_query_list"), x => split(x, "=")(0))
        )
      )
      .withColumn(
        "query_features", 
        transform(col("clean_query_list"), x=> split(x, "=")(1))
      )

    //df_chars.filter(col("clean_path").contains("moveStudent")).select("url_chars").show(truncate=false)

    /*
    val agent_nGram = new NGram().setN(7).setInputCol("agent").setOutputCol("agent_ngrams")
    val df_agent_ngram = agent_nGram.transform(df_prep)

    val agent_hashingTF = new HashingTF().setInputCol("agent_ngrams").setOutputCol("agent_hashes").setNumFeatures(32)

    val df_agent_hash = agent_hashingTF.transform(df_agent_ngram)
    */

    //df_ngram.filter(col("clean_path").contains("moveStudent")).select("url_chars", "ngrams").show(truncate=false)

    val url_hashingTF = new HashingTF()
      .setInputCol("url_features")
      .setOutputCol("url_hashes")
      .setNumFeatures(2040)

    val df_url_hash = url_hashingTF.transform(df_prep) 

    //df_url_hash.filter(col("clean_path").contains("moveStudent")).select("url_features", "url_hashes").show(truncate=false)

    val vector_assembler = new VectorAssembler()
      .setInputCols(
        Array(
          "fabstime", 
          "status", 
          "body_bytes_sent", 
          "query_len", 
          "url_hashes"
        )
      )
      .setOutputCol("features")

    val df_vectorized = vector_assembler.transform(df_url_hash)

    //df_vectorized.select("fabstime", "status", "body_bytes_sent", "url_hashes", "features").show(1, truncate=false)
    val isolationForest = new IsolationForest()
      .setNumEstimators(100)
      .setBootstrap(false)
      .setMaxSamples(256)
      .setMaxFeatures(1.0)
      .setFeaturesCol("features")
      .setPredictionCol("anomaly")
      .setScoreCol("outlierScore")
      .setContamination(0.02)
      .setRandomSeed(1)

    val isolationForestModel = isolationForest.fit(df_vectorized)


    isolationForestModel.write.overwrite.save("s3a://logs/output/5-detect-model/domain_category=" + curr_domain + "/")

    /*
    val isolationForestModel2 = IsolationForestModel.load()
    */

    val df_scores = isolationForestModel.transform(df_vectorized)
    
    df_scores.write
      .partitionBy("domain_category")
      .parquet("s3a://logs/output/5-detect-predict/")


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
      .csv("s3a://logs/output/log/5-detect/domain_category=" + curr_domain + "/")

    /*  
    val df2 = df_scores.withColumn("anomaly_bool", col("anomaly").cast(BooleanType))

    df2.select("fabstime", "status", "body_bytes_sent", "outlierScore", "anomaly_bool").write.format("es").mode("overwrite").save("anom")
    */

  }
}

