import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{StructType, StructField, StringType}
import java.text.SimpleDateFormat
import java.util.Date
import scala.sys.process._

object ScalaApp {
  def main(args: Array[String]): Unit = {
  
    val spark = SparkSession.builder.appName("98-template")
      .master("spark://spark-master:7077")
      .config("spark.hadoop.fs.s3a.endpoint", sys.env("S3_HOST"))
      .config("spark.hadoop.fs.s3a.access.key", sys.env("S3_USER"))
      .config("spark.hadoop.fs.s3a.secret.key", sys.env("S3_PASS"))
      .config("spark.hadoop.fs.s3a.path.style.access", "true")
      .getOrCreate()
  
    val start_time = new SimpleDateFormat("yy-MM-dd-HH_mm").format(new Date())





    val log = Seq(
      Row("Start", start_time),
      Row("End", new SimpleDateFormat("yy-MM-dd-HH_mm").format(new Date())),
      Row("Volume", s"${df.count()}, ${df.columns.length}")
    )
    
    // Schema
    val log_schema = StructType(Seq(
      StructField("Metric", StringType, true),
      StructField("Value", StringType, true)
    ))

    val df_log = spark.createDataFrame(spark.sparkContext.parallelize(log), log_schema)

    log_df.coalesce(1).write
      .option("header", "true")
      .csv("s3a://logs/output/log/98-template")
  }
}
