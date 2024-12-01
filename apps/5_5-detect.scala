// podman exec -it spark-master ./bin/spark-shell  --packages org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.526,com.linkedin.isolation-forest:isolation-forest_3.5.0_2.12:3.0.6 --repositories https://repo.maven.apache.org/maven2 --conf spark.hadoop.fs.s3a.endpoint=http://minio:9000 --conf spark.hadoop.fs.s3a.access.key=accesskey --conf spark.hadoop.fs.s3a.secret.key=secretkey --conf spark.hadoop.fs.s3a.path.style.access=true --conf spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem
//
// podman exec -it spark-master ./bin/spark-shell  --packages org.apache.hadoop:hadoop-aws:3.3.4,com.linkedin.isolation-forest:isolation-forest_3.5.0_2.12:3.0.6 --repositories https://repo.maven.apache.org/maven2 --conf spark.hadoop.fs.s3a.endpoint=http://minio:9000 --conf spark.hadoop.fs.s3a.access.key=accesskey --conf spark.hadoop.fs.s3a.secret.key=secretkey --conf spark.hadoop.fs.s3a.path.style.access=true --conf spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem

import com.linkedin.relevance.isolationforest._
import org.apache.spark.ml.feature.VectorAssembler
 
/**
  * Load and prepare data
  */
 
// Dataset from http://odds.cs.stonybrook.edu/shuttle-dataset/
val rawData = spark.read
  .format("csv")
  .option("comment", "#")
  .option("header", "false")
  .option("inferSchema", "true")
  .load("s3a://ifor/shuttle.csv")
 
val cols = rawData.columns
val labelCol = cols.last
 
val assembler = new VectorAssembler()
  .setInputCols(cols.slice(0, cols.length - 1))
  .setOutputCol("features")
val data = assembler
  .transform(rawData)
  .select(col("features"), col(labelCol).as("label"))
 
// scala> data.printSchema
// root
//  |-- features: vector (nullable = true)
//  |-- label: integer (nullable = true)
 
/**
  * Train the model
  */
 
val isolationForest = new IsolationForest()
  .setNumEstimators(100)
  .setBootstrap(false)
  .setMaxSamples(256)
  .setMaxFeatures(1.0)
  .setFeaturesCol("features")
  .setPredictionCol("predictedLabel")
  .setScoreCol("outlierScore")
  .setContamination(0.1)
  .setRandomSeed(1)
 
val isolationForestModel = isolationForest.fit(data)
 
/**
  * Score the training data
  */
 
val dataWithScores = isolationForestModel.transform(data)
 
// scala> dataWithScores.printSchema
// root
//  |-- features: vector (nullable = true)
//  |-- label: integer (nullable = true)
//  |-- outlierScore: double (nullable = false)
//  |-- predictedLabel: double (nullable = false)
