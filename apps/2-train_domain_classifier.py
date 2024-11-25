from datetime import datetime
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
)
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

current_date = datetime.now().strftime("%y-%m-%d-%H_%M")

df_known = (
    spark.read.parquet("s3a://logs/output/1-extract/")
    .select(
        "fabstime",
        "day_of_week",
        "body_bytes_sent",
        "clean_path",
        "clean_query_list",
        "domain_category",
        "method",
        "domain",
    )
    .filter(col("domain_category") != "other")
)

method_indexer = StringIndexer(inputCol="method", outputCol="method_index")
method_encoder = OneHotEncoder(inputCol="method_index", outputCol="method_onehot")

method_i_fit = method_indexer.fit(df_known)
df_known = method_i_fit.transform(df_known)

method_e_fit = method_encoder.fit(df_known)
df_known = method_e_fit.transform(df_known)

domain_indexer = StringIndexer(inputCol="domain", outputCol="domain_index")
domain_encoder = OneHotEncoder(inputCol="domain_index", outputCol="domain_onehot")

domain_i_fit = domain_indexer.fit(df_known)
df_known = domain_i_fit.transform(df_known)

domain_e_fit = domain_encoder.fit(df_known)
df_known = domain_e_fit.transform(df_known)

domain_i_fit.write().overwrite().save("s3a://logs/output/2-train_domain_classifier/" + current_date + "/metadata/domain/indexer")
domain_e_fit.write().overwrite().save("s3a://logs/output/2-train_domain_classifier/" + current_date + "/metadata/domain/encoder")

method_i_fit.write().overwrite().save("s3a://logs/output/2-train_domain_classifier/" + current_date + "/metadata/method/indexer")
method_e_fit.write().overwrite().save("s3a://logs/output/2-train_domain_classifier/" + current_date + "/metadata/method/encoder")


"""
from pyspark.ml.feature import StringIndexerModel, OneHotEncoderModel
domain_i_fit = StringIndexerModel.load("s3a://logs/output/2-train_domain_classifier/" + current_date + "/metadata/domain/indexer")
domain_e_fit = OneHotEncoderModel.load("s3a://logs/output/2-train_domain_classifier/" + current_date + "/metadata/domain/encoder")

df = domain_i_fit.transform(df)
df = domain_e_fit.transform(df)
"""


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
    "s3a://logs/output/2-train_domain_classifier/" + current_date + "/model_domain_classifier/"
)
#mc cp -r myminio/logs/models/24-11-16-03_25/domain_classifier/data myminio/logs/output/2-train_domain_classifier/24-11-16-03_25/model_domain_classifier/
#mc cp -r myminio/logs/models/24-11-16-03_25/domain_classifier/metadata myminio/logs/output/2-train_domain_classifier/24-11-16-03_25/model_domain_classifier/

#mc cp -r myminio/logs/metadata/ myminio/logs/output/2-train_domain_classifier/24-11-16-03_25/metadata/

predictions = lr_model.transform(test_data)

evaluator = MulticlassClassificationEvaluator(
    labelCol="domain_index", predictionCol="prediction", metricName="accuracy"
)
t_accuracy = evaluator.evaluate(predictions)
print(f"Test Accuracy: {t_accuracy}")


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
w_precision = evaluator_precision.evaluate(predictions)
print(f"Weighted Precision: {w_precision}")
# weightedPrecision: Precision considering class imbalance.

# Create the evaluator with weightedRecall
evaluator_recall = MulticlassClassificationEvaluator(
    labelCol="domain_index", predictionCol="prediction", metricName="weightedRecall"
)

# Evaluate the recall
w_recall = evaluator_recall.evaluate(predictions)
print(f"Weighted Recall: {w_recall}")
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
print(f"Test Accuracy: {t_accuracy}")
print(f"Accuracy: {accuracy}")
print(f"Weighted Precision: {w_precision}")
print(f"Weighted Recall: {w_recall}")
print(f"F1 Score: {f1_score}")

log = [
    ("Start", current_date),
    ("End", datetime.now().strftime("%y-%m-%d-%H_%M")),
    ("Test Accuracy", t_accuracy),
    ("Accuracy", accuracy),
    ("Weighted Precision", w_precision),
    ("Weighted Recall", w_recall),
    ("F1 Score", f1_score),
]

df_log = spark.createDataFrame(log, ["Metric", "Value"])

df_log.write.csv("s3a://logs/output/2-train_domain_classifier/" + current_date + "/log.csv", header=True)
