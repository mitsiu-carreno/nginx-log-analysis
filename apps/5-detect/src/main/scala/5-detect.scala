import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions._
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

    val df = spark.read
      .parquet("s3a://logs/output/4-domain_ground_truth/")
      .filter(col("domain_category") === "canvas.ieec.mx")
    
    val vector_assembler = new VectorAssembler()
      .setInputCols(Array("fabstime", "status", "body_bytes_sent"))
      .setOutputCol("features")

    val df_test = vector_assembler.transform(df)
      .select(col("domain_category"), col("fabstime"), col("status"), col("body_bytes_sent") ,col("features"))

    val isolationForest = new IsolationForest()
      .setNumEstimators(100)
      .setBootstrap(false)
      .setMaxSamples(256)
      .setMaxFeatures(1.0)
      .setFeaturesCol("features")
      .setPredictionCol("anomaly")
      .setScoreCol("outlierScore")
      .setContamination(0.1)
      .setRandomSeed(1)

    val isolationForestModel = isolationForest.fit(df_test)

    val df_scores = isolationForestModel.transform(df_test)
    
    // df_scores.filter(col("anomaly") === 1).show(truncate=false)
    // df_scores.filter(col("anomaly") === 1).count
    
    df_scores.write
      .partitionBy("domain_category")
      .parquet("s3a://logs/output/5-detect")

    val log = Seq(
      Row("Start", start_time),
      Row("End", new SimpleDateFormat("yy-MM-dd-HH_mm").format(new Date())),
      Row("domain_category", "canvas.ieec.mx"),
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
  }
}
