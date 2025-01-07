from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import (
    col,
    udf,
)
from sklearn.ensemble import IsolationForest
import pandas as pd
from datetime import datetime
import os

spark = SparkSession.builder.appName("3-predict_domain") \
    .master("spark://spark-master:7077") \
    .config("spark.hadoop.fs.s3a.endpoint", os.getenv("S3_HOST")) \
    .config("spark.hadoop.fs.s3a.access.key", os.getenv("S3_USER")) \
    .config("spark.hadoop.fs.s3a.secret.key", os.getenv("S3_PASS")) \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .getOrCreate()

start_time = datetime.now().strftime("%y-%m-%d-%H_%M")

train_date = "24-11-16-03_25"

df = spark.read.parquet("s3a://logs/output/4-domain_ground_truth/").filter(
    col("domain_category") == "canvas.ieec.mx"
)



df_pd = df.toPandas()
iso_forest = IsolationForest()
df_reshaped = df_pd[["body_bytes_sent"]].values.reshape(-1, 1)
iso_forest.fit(df_reshaped)

@pandas_udf(IntegerType())
def isolation_forest_udf(pdf: pd.Series) -> pd.Series:
    pdf_reshaped = pdf.values.reshape(-1, 1)
    predictions = iso_forest.predict(pdf_reshaped)
    return pd.Series(predictions)

df_with_outliers = df.withColumn("outlier", isolation_forest_udf(df["body_bytes_sent"]))

print("_"*200)
print(df_with_outliers.filter(col("outlier") == 1).count())
