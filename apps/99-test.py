from pyspark.sql import SparkSession
import os

# Define the two MinIO paths
path1 = "s3a://logs/output/1-extract/"
path2 = "s3a://logs/output/2-predict_domain/"

spark = SparkSession.builder.appName("99-test") \
    .master("spark://spark-master:7077") \
    .config("spark.hadoop.fs.s3a.endpoint", os.getenv("S3_HOST")) \
    .config("spark.hadoop.fs.s3a.access.key", os.getenv("S3_USER")) \
    .config("spark.hadoop.fs.s3a.secret.key", os.getenv("S3_PASS")) \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .getOrCreate()



# Read the data from both paths
df1 = spark.read.format("parquet").load(path1)  # or use another format like "csv"
df2 = spark.read.format("parquet").load(path2)

df_combined = df1.unionByName(df2, allowMissingColumns=True)

# Show the resulting DataFrame
df_combined.printSchema()


"""
from pyspark.sql import SparkSession

from pyspark.ml.feature import StringIndexerModel, OneHotEncoderModel

spark = SparkSession.builder.appName("1-extract").getOrCreate()

df = spark.read.parquet("s3a://logs/output/extract").limit(2)

df.printSchema()

df.show(truncate=False)


domain_i_fit = StringIndexerModel.load("s3a://logs/metadata/domain/indexer")


# The `labels` attribute contains the original category names, indexed in the order of the `domain_index`
domain_labels = domain_i_fit.labels

print(domain_labels)

# Create a UDF to map the predicted index back to the domain name
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType


# UDF to map index back to category name
def index_to_domain(index):
    return domain_labels[int(index)]


# Register the UDF
index_to_domain_udf = udf(index_to_domain, StringType())

# Apply the UDF to create the new column 'prediction_domain'
predictions_with_domain = predictions.withColumn(
    "prediction_domain", index_to_domain_udf(predictions["prediction"])
)

# Show the resulting DataFrame
predictions_with_domain.select("prediction", "prediction_domain").show()
"""
