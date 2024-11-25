from pyspark.sql import SparkSession
import os
from datetime import datetime
from pyspark.sql.functions import (
    col,
)

spark = SparkSession.builder.appName("4-domain_ground_truth") \
    .master("spark://spark-master:7077") \
    .config("spark.hadoop.fs.s3a.endpoint", os.getenv("S3_HOST")) \
    .config("spark.hadoop.fs.s3a.access.key", os.getenv("S3_USER")) \
    .config("spark.hadoop.fs.s3a.secret.key", os.getenv("S3_PASS")) \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .getOrCreate()

start_time = datetime.now().strftime("%y-%m-%d-%H_%M")

known_path = "s3a://logs/output/1-extract/"
unknown_path = "s3a://logs/output/3-predict_domain/"

df_known = spark.read.parquet(known_path).filter(
    col("domain_category") != "other"
)

df_unknown = spark.read.parquet(unknown_path).select(
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
	"prediction_domain",
).withColumnRenamed("prediction_domain", "domain_category")

df_final = df_known.unionByName(df_unknown, allowMissingColumns=False)

df_final.printSchema()

df_final.write.partitionBy("domain_category").parquet("s3a://logs/output/4-domain_ground_truth")

df_final.groupBy("domain_category").count().show(
    500, truncate=False
)

print(f"{df_final.count()}, {len(df_final.columns)}")

log = [
    ("Start", start_time),
    ("End", datetime.now().strftime("%y-%m-%d-%H_%M")),
    ("Volume", f"{df_final.count()}, {len(df_final.columns)}"),
]

df_log = spark.createDataFrame(log, ["Metric", "Value"])

df_log.coalesce(1).write.csv("s3a://logs/output/log/4-domain_ground_truth", header=True)


