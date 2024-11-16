from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    regexp_extract,
    col,
    udf,
    isnan,
    when,
    trim,
    hour,
    minute,
    dayofweek,
)
from pyspark.sql.types import IntegerType, StringType, ArrayType, TimestampType
import urllib.parse
from datetime import datetime
import time
import os
import ast

spark = SparkSession.builder.appName("1-extract") \
    .master("spark://spark-master:7077") \
    .config("spark.hadoop.fs.s3a.endpoint", os.getenv("S3_HOST")) \
    .config("spark.hadoop.fs.s3a.access.key", os.getenv("S3_USER")) \
    .config("spark.hadoop.fs.s3a.secret.key", os.getenv("S3_PASS")) \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .getOrCreate()

df_raw = spark.read.text("s3a://logs/input/**")

log_columns = [
    "remote_addr",
    "remote_usr",
    "date_time",
    "method",
    "req_uri",
    "http_ver",
    "status",
    "body_bytes_sent",
    "http_referer",
    "user_agent",
    "gzip_ratio",
]

log_pattern = r"^((?:\d{1,3}\.){3}\d{1,3}|[0-9a-fA-F:]+) - (-|[^\s]+) \[([^\]]+)\] \"([A-Z]+) ([^ ]+) (HTTP/\d\.\d)\" (\d{3}) (\d+) \"([^\"]*)\" \"([^\"]*)\"(?:\s+(\d+))?"

df_parsed = df_raw.select(
    *[
        regexp_extract(col("value"), log_pattern, i).alias(log_columns[i - 1])
        for i in range(1, len(log_columns) + 1)
    ]
)

df_parsed = df_parsed.selectExpr(
    "remote_addr",
    "remote_usr",
    "date_time",
    "substring(date_time, 1, 11) as date",
    "substring(date_time, 13, 8) as time",
    "method",
    "req_uri",
    "http_ver",
    "status",
    "body_bytes_sent",
    "http_referer",
    "user_agent",
    "gzip_ratio",
)

df_parsed = df_parsed.withColumn("status", df_parsed["status"].cast(IntegerType()))
df_parsed = df_parsed.withColumn(
    "body_bytes_sent", df_parsed["body_bytes_sent"].cast(IntegerType())
)

df_parsed = df_parsed.drop("gzip_ratio")


# Change null and empty to nan
def to_null(c):
    return when(~(col(c).isNull() | isnan(col(c)) | (trim(col(c)) == "")), col(c))


df_parsed = df_parsed.select([to_null(c).alias(c) for c in df_parsed.columns]).na.drop()


# User defined functions
def decode_uri(uri):
    return urllib.parse.unquote(uri) if uri is not None else None


def get_path(uri):
    return urllib.parse.urlparse(uri).path if uri is not None else None


def get_query_list(uri):
    if uri is None:
        return []
    query = urllib.parse.urlparse(uri).query
    return [f"{k}={v}" for k, v in urllib.parse.parse_qsl(query)]


def get_domain(referer):
    netloc = urllib.parse.urlparse(referer).netloc
    return netloc if netloc not in (None, "", "-") else "Unknown"
    # return urllib.parse.urlparse(referer).netloc if referer not in (None, "-") else "Unknown"


def parse_date(date_str):
    return (
        datetime.strptime(date_str, "%d/%b/%Y:%H:%M:%S +0000")
        if date_str is not None
        else None
    )


def to_unix_timestamp(dt):
    return time.mktime(dt.timetuple()) if dt is not None else None


# Register UDF's
decode_uri_udf = udf(decode_uri, StringType())
get_path_udf = udf(get_path, StringType())
get_query_list_udf = udf(get_query_list, ArrayType(StringType()))
get_domain_udf = udf(get_domain, StringType())
parse_date_udf = udf(parse_date, TimestampType())
to_unix_timestamp_udf = udf(to_unix_timestamp, StringType())

df = df_parsed
df = df.withColumn("dec_req_uri", decode_uri_udf(col("req_uri")))
df = df.withColumn("clean_path", get_path_udf(col("dec_req_uri")))
df = df.withColumn("clean_query_list", get_query_list_udf(col("dec_req_uri")))
df = df.withColumn("domain", get_domain_udf(col("http_referer")))
df = df.withColumn("fdate_time", parse_date_udf(col("date_time")))
df = df.withColumn("dateunixtimest", to_unix_timestamp_udf(col("fdate_time")))
df = df.withColumn(
    "fabstime", (hour(col("fdate_time")) + minute(col("fdate_time")) / 60.0)
)

df = df.withColumn("day_of_week", dayofweek(col("fdate_time")))


domains_str = os.getenv("DOMAINS")

registered_domains = ast.literal_eval(domains_str)

df = df.withColumn(
    "domain_category",
    when(col("domain").isin(*registered_domains), col("domain")).otherwise("other"),
)

df.write.partitionBy("domain_category").parquet("s3a://logs/output/1-extract/")


df.groupBy("domain_category").count().show(
    500, truncate=False
)
