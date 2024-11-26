from datetime import datetime
from pyspark.sql import SparkSession
import os

spark = SparkSession.builder.appName("2-train_domain_classifier") \
    .master("spark://spark-master:7077") \
    .config("spark.hadoop.fs.s3a.endpoint", os.getenv("S3_HOST")) \
    .config("spark.hadoop.fs.s3a.access.key", os.getenv("S3_USER")) \
    .config("spark.hadoop.fs.s3a.secret.key", os.getenv("S3_PASS")) \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .getOrCreate()

current_date = datetime.now().strftime("%y-%m-%d-%H_%M")

df = spark.read.parquet("s3a://logs/output/4-domain_ground_truth/")

df.groupBy("date").count().coalesce(1).write.csv("s3a://logs/output/log/50-metadata/byDate", header=True)


df.groupBy("date", "domain_category").count().coalesce(1).write.csv("s3a://logs/output/log/50-metadata/byDateDomain", header=True)

### Playground
from pyspark.sql.functions import explode, col, length, countDistinct, min, max, avg, split

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType

df_exploded = df.withColumn("path_words", split(col("clean_path"), "/"))

# Count unique words
df_exploded = df_exploded.withColumn("word", explode(col("path_words")))
unique_words_count = df_exploded.select("word").distinct().count()
print(f"Unique words count: {unique_words_count}")

# 2. Get min, max, and average characters per word
df_with_word_length = df_exploded.withColumn("word_length", length(col("word")))

# Min, Max, and Average word lengths
min_length = df_with_word_length.select(min("word_length")).collect()[0][0]
max_length = df_with_word_length.select(max("word_length")).collect()[0][0]
avg_length = df_with_word_length.select(avg("word_length")).collect()[0][0]

print(f"Minimum word length: {min_length}")
print(f"Maximum word length: {max_length}")
print(f"Average word length: {avg_length}")

# Calculating the standard deviation of word lengths
stddev_length = df_with_word_length.agg(F.stddev("word_length")).collect()[0][0]
print(f"Standard Deviation of word length: {stddev_length}")

# Calculating the length of each array (number of words)
df_with_array_length = df_exploded.withColumn("array_length", F.size("path_words"))

# Calculating min, max, and average length of the arrays
min_array_length = df_with_array_length.agg(F.min("array_length")).collect()[0][0]
max_array_length = df_with_array_length.agg(F.max("array_length")).collect()[0][0]
avg_array_length = df_with_array_length.agg(F.avg("array_length")).collect()[0][0]

# Calculating standard deviation of array lengths
stddev_array_length = df_with_array_length.agg(F.stddev("array_length")).collect()[0][0]

print(f"Min array length: {min_array_length}")
print(f"Max array length: {max_array_length}")
print(f"Average array length: {avg_array_length}")
print(f"Standard Deviation of array length: {stddev_array_length}")

###End-playground

log = [
    ("Start", current_date),
    ("End", datetime.now().strftime("%y-%m-%d-%H_%M")),
    ("Unique words count", unique_words_count),
    ("Minimum word length", min_length),
    ("Maximum word length", max_length),
    ("Average word length", avg_length),
    ("Standard Deviation of word length", stddev_length),
    ("Min array length", min_array_length),
    ("Max array length", max_array_length),
    ("Average array length", avg_array_length),
    ("Standard Deviation of array length", stddev_array_length),
]

df_log = spark.createDataFrame(log, ["Metric", "Value"])

df_log.coalesce(1).write.csv("s3a://logs/output/log/50-metadata/general", header=True)
