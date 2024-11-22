from pyspark.ml.classification import LogisticRegressionModel
from pyspark.sql import SparkSession
from pyspark.ml.feature import (
    NGram,
    HashingTF,
    VectorAssembler,
    StringIndexerModel,
    OneHotEncoderModel,
)
from pyspark.sql.functions import (
    col,
    split,
    concat,
    transform,
    udf,
)
from pyspark.sql.types import StringType
import os


spark = SparkSession.builder.appName("3-predict_domain") \
    .master("spark://spark-master:7077") \
    .config("spark.hadoop.fs.s3a.endpoint", os.getenv("S3_HOST")) \
    .config("spark.hadoop.fs.s3a.access.key", os.getenv("S3_USER")) \
    .config("spark.hadoop.fs.s3a.secret.key", os.getenv("S3_PASS")) \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .getOrCreate()

df_unknown_domains = spark.read.parquet("s3a://logs/output/1-extract/").filter(
    col("domain_category") == "other"
)

df_unknown_domains = df_unknown_domains.withColumn(
    "path_characters", split(col("clean_path"), "")
)

ngram = NGram(n=9, inputCol="path_characters", outputCol="path_ngrams")

df_unknown_domains = ngram.transform(df_unknown_domains)

df_unknown_domains = df_unknown_domains.withColumn(
    "url_features",
    concat(
        col("path_ngrams"),
        transform(col("clean_query_list"), lambda x: split(x, "=")[0]),
    ),
)

hashingTF = HashingTF(
    inputCol="url_features", outputCol="hash_url_features", numFeatures=16384
)

df_unknown_domains = hashingTF.transform(df_unknown_domains)

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

df_unknown_domains = vector_assembler.transform(df_unknown_domains)

lr_model = LogisticRegressionModel.load(
    "s3a://logs/models/24-11-15-03_04/domain_classifier"
)

predictions = lr_model.transform(df_unknown_domains)

domain_i_fit = StringIndexerModel.load("s3a://logs/metadata/domain/indexer")
domain_e_fit = OneHotEncoderModel.load("s3a://logs/metadata/domain/encoder")

domain_labels = domain_i_fit.labels
print(domain_labels)

predictions.show(50, truncate=False)


def index_to_domain(index):
    return domain_labels[int(index)]


index_to_domain_udf = udf(index_to_domain, StringType())

predictions = predictions.withColumn(
    "prediction_domain", index_to_domain_udf(predictions["prediction"])
)

df_final = predictions.select(
    "remote_addr",
    "remote_usr",
    "date_time",
    "date",
    "time",
    "method",
    "req_uri",
    "http_ver",
    "status",
    "body_bytes_sent",
    "http_referer",
    "user_agent",
    "dec_req_uri",
    "clean_path",
    "clean_query_list",
    "domain",
    "fdate_time",
    "dateunixtimest",
    "fabstime",
    "day_of_week",
    "method_index",
    "method_onehot",
    "domain_index",
    "domain_onehot",
    "probability",
    "prediction",
    "prediction_domain",
)

df_final.write.paritionBy("predicton_domain").parquet(
    "s3a://logs/output/2-predict_domain/"
)

predictions.groupBy("prediction_domain").count().show(500, truncate=False)

