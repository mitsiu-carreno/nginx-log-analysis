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

val df = spark.read.parquet("s3a://logs/output/4-domain_ground_truth/").filter(col("domain_category") === curr_domain).select("method","status","body_bytes_sent","http_referer","user_agent","dec_req_uri","clean_path","clean_query_list","domain","fdate_time","fabstime","day_of_week","domain_category")


val df_prep = df
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
        "url_features_string",
         concat_ws(" ", col("url_features_regex"))
      )


import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.{Tokenizer, Doc2VecModel}
import com.johnsnowlabs.nlp.EmbeddingsFinisher
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler().setInputCol("url_features_string").setOutputCol("document")
val tokenizer = new Tokenizer().setInputCols(Array("document")).setOutputCol("token")

val embeddings = Doc2VecModel.pretrained().setInputCols("token").setOutputCol("embeddings")

val embeddingsFinisher = new EmbeddingsFinisher().setInputCols("embeddings").setOutputCols("finished_embeddings").setOutputAsVector(true)

val pipeline = new Pipeline().setStages(Array(documentAssembler,tokenizer,embeddings,embeddingsFinisher))

val result = pipeline.fit(df_prep).transform(df_prep)

result.selectExpr("url_features_string", "explode(finished_embeddings) as result").show(3, false)

val df_url_hash = result.selectExpr("fabstime", "status", "body_bytes_sent", "clean_path", "explode(finished_embeddings) as url_hashes")

val vector_assembler = new VectorAssembler().setInputCols(Array("fabstime","status","body_bytes_sent","url_hashes")).setOutputCol("features")


val df_vectorized = vector_assembler.transform(df_url_hash)

val isolationForest = new IsolationForest().setNumEstimators(100).setBootstrap(false).setMaxSamples(256).setMaxFeatures(1.0).setFeaturesCol("features").setPredictionCol("anomaly").setScoreCol("outlierScore").setContamination(0.02).setRandomSeed(1)

val isolationForestModel = isolationForest.fit(df_vectorized)

val df_scores = isolationForestModel.transform(df_vectorized)

df_scores.filter(col("anomaly")=== 1).select("clean_path").show(false)
