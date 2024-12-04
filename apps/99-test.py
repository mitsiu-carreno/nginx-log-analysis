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


"""
import seaborn as sns
import matplotlib.pyplot as plt
import boto3
from io import BytesIO

# 1. Generate a Seaborn plot
# Example: A simple heatmap
data = sns.load_dataset("flights")
pivot_data = data.pivot(index="month", columns="year", values="passengers")
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_data, annot=True, fmt="g", cmap='coolwarm')

# 2. Save the plot to a BytesIO object (in-memory)
image_stream = BytesIO()
plt.savefig(image_stream, format='png', bbox_inches='tight')  # Save to in-memory stream
image_stream.seek(0)  # Rewind the BytesIO stream to the beginning

# 3. Connect to MinIO using boto3
minio_client = boto3.client(
    's3', 
    endpoint_url='https://your-minio-endpoint',  # Replace with your MinIO endpoint
    aws_access_key_id='your-access-key',        # Replace with your MinIO access key
    aws_secret_access_key='your-secret-key',    # Replace with your MinIO secret key
    region_name='us-east-1'                      # Replace with your region (if needed)
)

# 4. Upload the image to MinIO (replace 'your-bucket' with your bucket name)
bucket_name = 'your-bucket-name'  # Replace with your MinIO bucket name
file_name = 'output/plots/seaborn_heatmap.png'  # Path to save the file in MinIO

minio_client.upload_fileobj(image_stream, bucket_name, file_name)

# Close the stream after uploading
image_stream.close()

print(f"Plot successfully uploaded to MinIO at {bucket_name}/{file_name}")
"""

"""
# Read the data from both paths
df1 = spark.read.format("parquet").load(path1)  # or use another format like "csv"
df1.printSchema()
print("=======================")
df2 = spark.read.format("parquet").load(path2)

df_combined = df1.unionByName(df2, allowMissingColumns=True)

# Show the resulting DataFrame
df_combined.printSchema()
"""

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
