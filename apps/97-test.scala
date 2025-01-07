import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{NGram, HashingTF}
import org.apache.spark.sql.functions._

val data = spark.createDataFrame(Seq(  (1, "/api/v1/student/1234"),  (2, "/api/v1/student/1233"),  (3, "/api/v1/teacher/5678"))).toDF("id", "uri")

val dataWithChars = data.withColumn("uri_chars", split(col("uri"), ""))

val nGram = new NGram().setN(4).setInputCol("uri_chars").setOutputCol("ngrams")

val ngramData = nGram.transform(dataWithChars)

val hashingTF = new HashingTF().setInputCol("ngrams").setOutputCol("features").setNumFeatures(100)

val hashedData = hashingTF.transform(ngramData)


// =========================MINHASH==================================================



import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{MinHashLSH, HashingTF}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions._

val spark = SparkSession.builder.appName("MinHashLSH Example").getOrCreate()

// Sample data with URI paths
val data = spark.createDataFrame(Seq(
  (1, "/api/v1/student/1234"),
  (2, "/api/v1/student/1233"),
  (3, "/api/v1/teacher/5678")
)).toDF("id", "uri")

// Step 1: Convert URI paths into an array of characters (this will be treated as tokens)
val dataWithChars = data.withColumn("uri_chars", split(col("uri"), ""))

// Step 2: Apply HashingTF to the array of characters to create feature vectors
val hashingTF = new HashingTF().setInputCol("uri_chars").setOutputCol("raw_features").setNumFeatures(100)
val featurizedData = hashingTF.transform(dataWithChars)

// Step 3: Apply MinHashLSH on the features to compute hash signatures
val minHashLSH = new MinHashLSH()
  .setInputCol("raw_features")
  .setOutputCol("hashes")
  .setNumHashTables(5)  // Number of hash tables, tune this for better accuracy/performance

val hashedData = minHashLSH.fit(featurizedData).transform(featurizedData)
