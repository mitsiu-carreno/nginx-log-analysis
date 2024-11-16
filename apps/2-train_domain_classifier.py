from datetime import datetime
from pyspark.ml.feature import (
    NGram,
    HashingTF,
    VectorAssembler,
)
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import (
    col,
    split,
    concat,
    transform,
)
import os

spark = SparkSession.builder.appName("2-train_domain_classifier") \
    .master("spark://spark-master:7077") \
    .config("spark.hadoop.fs.s3a.endpoint", os.getenv("S3_HOST")) \
    .config("spark.hadoop.fs.s3a.access.key", os.getenv("S3_USER")) \
    .config("spark.hadoop.fs.s3a.secret.key", os.getenv("S3_PASS")) \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .getOrCreate()

df_known = (
    spark.read.parquet("s3a://logs/output/1-extract/")
    .select(
        "fabstime",
        "day_of_week",
        "method_onehot",
        "body_bytes_sent",
        "clean_path",
        "clean_query_list",
        "domain_index",
        "domain_category",
    )
    .filter(col("domain_category") != "other")
)

df_known = df_known.withColumn("path_characters", split(col("clean_path"), ""))

ngram = NGram(n=9, inputCol="path_characters", outputCol="path_ngrams")

df_known = ngram.transform(df_known)

df_known = df_known.withColumn(
    "url_features",
    concat(
        col("path_ngrams"),
        transform(col("clean_query_list"), lambda x: split(x, "=")[0]),
    ),
)

hashingTF = HashingTF(
    inputCol="url_features", outputCol="hash_url_features", numFeatures=16384
)

df_known = hashingTF.transform(df_known)

vector_assembler = VectorAssembler(
    inputCols=[
        # "levenshtein_distance",
        "fabstime",
        "day_of_week",
        "method_onehot",
        "body_bytes_sent",
        "hash_url_features",
    ],
    outputCol="features",
)

df = vector_assembler.transform(df_known)

lr = LogisticRegression(
    featuresCol="features", labelCol="domain_index", family="multinomial", maxIter=100
)

train_data, test_data = df.randomSplit([0.8, 0.2], seed=123)

lr_model = lr.fit(train_data)

lr_model.save(
    "s3a://logs/models/"
    + datetime.now().strftime("%y-%m-%d-%H_%M")
    + "/domain_classifier"
)

predictions = lr_model.transform(test_data)

evaluator = MulticlassClassificationEvaluator(
    labelCol="domain_index", predictionCol="prediction", metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)
print(f"Test Accuracy: {accuracy}")


# Evaluate the model
evaluator = MulticlassClassificationEvaluator(
    labelCol="domain_index", predictionCol="prediction"
)
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}")

# Precision, recall, F1-score, etc.
evaluator_precision = MulticlassClassificationEvaluator(
    labelCol="domain_index", predictionCol="prediction", metricName="weightedPrecision"
)
precision = evaluator_precision.evaluate(predictions)
print(f"Weighted Precision: {precision}")
# weightedPrecision: Precision considering class imbalance.

# Create the evaluator with weightedRecall
evaluator_recall = MulticlassClassificationEvaluator(
    labelCol="domain_index", predictionCol="prediction", metricName="weightedRecall"
)

# Evaluate the recall
recall = evaluator_recall.evaluate(predictions)
print(f"Weighted Recall: {recall}")
# weightedRecall: Recall considering class imbalance.

# Create the evaluator with f1 score
evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="domain_index", predictionCol="prediction", metricName="f1"
)

# Evaluate the F1 score
f1_score = evaluator_f1.evaluate(predictions)
print(f"F1 Score: {f1_score}")
# f1: F1 score, which is the harmonic mean of precision and recall.

df_known.printSchema()
print(f"Test Accuracy: {accuracy}")
print(f"Accuracy: {accuracy}")
print(f"Weighted Precision: {precision}")
print(f"Weighted Recall: {recall}")
print(f"F1 Score: {f1_score}")
