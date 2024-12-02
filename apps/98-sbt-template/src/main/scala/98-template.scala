import org.apache.spark.sql.SparkSession
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
      ("Start", start_time),
      ("End", new SimpleDateFormat("yy-MM-dd-HH_mm").format(new Date())),
    )

    import spark.implicits._

    val log_df = log.toDF("Metric", "Value")

    log_df.coalesce(1).write
      .option("header", "true")
      .csv("s3a://logs/output/log/98-template")
  }
}
