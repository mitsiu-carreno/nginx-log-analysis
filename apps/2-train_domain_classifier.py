from datetime import datetime
from pyspark.ml.feature import (
    NGram,
    HashingTF,
    VectorAssembler,
    StringIndexerModel,
    OneHotEncoderModel,
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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import boto3
from io import BytesIO


spark = SparkSession.builder.appName("2-train_domain_classifier") \
    .master("spark://spark-master:7077") \
    .config("spark.hadoop.fs.s3a.endpoint", os.getenv("S3_HOST")) \
    .config("spark.hadoop.fs.s3a.access.key", os.getenv("S3_USER")) \
    .config("spark.hadoop.fs.s3a.secret.key", os.getenv("S3_PASS")) \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .getOrCreate()

current_date = datetime.now().strftime("%y-%m-%d-%H_%M")

s3_client = boto3.client(
    's3',
    endpoint_url= os.getenv("S3_HOST"),
    aws_access_key_id= os.getenv("S3_USER"),
    aws_secret_access_key= os.getenv("S3_PASS"),
)

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

domain_i_fit = StringIndexerModel.load("s3a://logs/output/1-extract-metadata/domain/indexer")
#domain_e_fit = OneHotEncoderModel.load("s3a://logs/output/1-extract-metadata/domain/encoder")

method_i_fit = StringIndexerModel.load("s3a://logs/output/1-extract-metadata/method/indexer")
method_e_fit = OneHotEncoderModel.load("s3a://logs/output/1-extract-metadata/method/encoder")

df_known = domain_i_fit.transform(df_known)
#df_known = domain_e_fit.transform(df_known)

df_known = method_i_fit.transform(df_known)
df_known = method_e_fit.transform(df_known)




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

# Confussion matrix

total_count = predictions.count()

confusion_matrix = predictions.groupBy("domain_index", "prediction").count()

confusion_matrix = confusion_matrix.select(
    col("domain_index").cast("int").alias("domain_index"),
    col("prediction").cast("int").alias("prediction"),
    col("count"),
    (confusion_matrix["count"] / total_count * 100).alias("percentage")
)

confusion_matrix.coalesce(1).write.csv("s3a://logs/output/2-train_domain_classifier/" + current_date + "/confusion_matrix_raw", header=True)

confusion_matrix_pd = confusion_matrix.toPandas()

##### Count Confussion matrix

confusion_matrix_count_pivot = confusion_matrix_pd.pivot(index='domain_index', columns='prediction', values='count').fillna(0)

plt.figure(figsize=(15, 12))
sns.heatmap(confusion_matrix_count_pivot, annot=True, fmt="g", cmap="Blues", xticklabels=True, yticklabels=True)
plt.title('Matriz de confusión')
plt.xlabel('Dominio asignado')
plt.ylabel('Dominio real')

count_matrix = BytesIO()

plt.savefig(count_matrix, format='png', bbox_inches='tight')

count_matrix.seek(0) #rewind stream to the beginning

s3_client.upload_fileobj(count_matrix, "logs", "output/2-train_domain_classifier/" + current_date + "/confusion_matrix_count.png")

count_matrix.close()

plt.clf()

##### Percentage Confussion matrix

confusion_matrix_percen_pivot = confusion_matrix_pd.pivot(index='domain_index', columns='prediction', values='percentage').fillna(0)


plt.figure(figsize=(15, 12))
sns.heatmap(confusion_matrix_percen_pivot, annot=True, fmt=".1%", cmap="Blues", xticklabels=True, yticklabels=True)
plt.title('Matriz de confusión')
plt.xlabel('Dominio asignado')
plt.ylabel('Dominio real')

percen_matrix = BytesIO()

plt.savefig(percen_matrix, format='png', bbox_inches='tight')

percen_matrix.seek(0) #rewind stream to the beginning

s3_client.upload_fileobj(percen_matrix, "logs", "output/2-train_domain_classifier/" + current_date + "/confusion_matrix_percen.png")

percen_matrix.close()

plt.clf()

percen_matrix.close()

### Evaluate model

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

df_log.coalesce(1).write.csv("s3a://logs/output/log/2-train_domain_classifier/" + current_date, header=True)
