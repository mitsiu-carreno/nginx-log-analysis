import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{NGram, VectorAssembler, HashingTF, MinHashLSH}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType, DoubleType}
import com.linkedin.relevance.isolationforest._
import java.text.SimpleDateFormat
import java.util.Date
import scala.sys.process._
import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions._


var curr_domain: String = "acuerdo286.designa.mx"
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


val df_prep = df      
  .withColumn(
    "es_time", 
    date_format(col("fdate_time"), "yyyy-MM-dd'T'HH:mm:ss.SSSZ")
  )
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
    "url_features_regex",
    expr("""
      transform(
        url_features,
             x -> regexp_replace(
              x,
               '(?:^[a-fA-F0-9]{24}$)|(?:^\\d+$)',
              CASE
                WHEN x RLIKE '^[a-fA-F0-9]{24}$' THEN 'MONGOID'
                WHEN x RLIKE '^\\d+$' THEN 'DIGIT'
                ELSE X
              END
            )
      )
    """)
  )
  .withColumn(
    "query_features",
    expr("transform(clean_query_list, x -> substr(x, instr(x, '=') + 1, length(x)))")
  )

val url_hashingTF = new HashingTF()
  .setInputCol("url_features_regex")
  .setOutputCol("url_hashes")
  .setNumFeatures(512)

val df_url_hash = url_hashingTF.transform(df_prep)

df_url_hash
  .filter(
    col("clean_path").contains("moveStudent")
  )
  .select("url_features_regex", "url_hashes")
  .show(truncate=false)

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

val isolationForest = new IsolationForest()
  .setNumEstimators(1000)
  .setBootstrap(false)
  .setMaxSamples(512)
  .setMaxFeatures(1.0)
  .setFeaturesCol("features")
  .setPredictionCol("anomaly")
  .setScoreCol("outlierScore")
  .setContamination(0.05)
  .setContaminationError(0.01)
  .setRandomSeed(1)

val isolationForestModel = isolationForest.fit(df_vectorized)

isolationForestModel
  .write
  .overwrite
  .save("s3a://logs/output/5-detect-model/domain_category=" + curr_domain + "/")

val df_scores = isolationForestModel.transform(df_vectorized)

val df2 = df_scores.withColumn("anomaly_bool", col("anomaly").cast(BooleanType))

/*
df_scores
  .filter(col("clean_path").contains("ButSkGqO"))
  .groupBy("anomaly").count().show()
df_scores
  .filter(col("clean_path").contains("ButSkGqO") && col("anomaly") === 1)
  .limit(1)
  .union(
    df_scores
      .filter(col("clean_path").contains("ButSkGqO") && col("anomaly") === 0)
      .limit(1)
  )
  .show(truncate=false)
*/

df2.select(
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
  "domain_category", 
  "es_time", 
  "query_len", 
  "outlierScore",
  "anomaly_bool")
  .write  
  .parquet("s3a://logs/output/5-detect-predict/domain_category=" + curr_domain + "/")
