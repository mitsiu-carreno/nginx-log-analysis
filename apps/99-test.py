from dotenv import load_dotenv
import os
import ast

# Load environment variables from the .env file
#load_dotenv('config.env')

# Read the list from the environment variable
my_list_str = os.getenv('DOMAINS')

print(os.environ)

# Convert the string representation of the list into a Python list
my_list = ast.literal_eval(my_list_str)  # safely converts string to list

# Now you can use my_list in your Spark job
for e in my_list:
    print(e)
#print(my_list_str)

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
