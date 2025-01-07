from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import IntegerType
from sklearn.ensemble import IsolationForest
import pandas as pd

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Isolation Forest Anomaly Detection") \
    .getOrCreate()

# Sample data with 5 features
data = [
    (1, 3.0, 2.0, 5.0, 10.0, 100.0),  # normal data
    (2, 4.0, 3.0, 6.0, 12.0, 110.0),  # normal data
    (3, 3.5, 2.5, 5.5, 11.0, 105.0),  # normal data
    (4, 1000000.0, 1000000.0, 1000000.0, 1000000.0, 1000000.0),  # outlier
    (5, 5.0, 3.5, 7.0, 14.0, 120.0)  # normal data
]
columns = ["id", "feature1", "feature2", "feature3", "feature4", "feature5"]

# Create Spark DataFrame
df = spark.createDataFrame(data, columns)

# Train IsolationForest on the entire dataset using multiple features
df_pd = df.toPandas()  # Convert Spark DataFrame to Pandas DataFrame for model training
iso_forest = IsolationForest(contamination=0.2)

# Use all 5 features for training
df_reshaped = df_pd[['feature1', 'feature2', 'feature3', 'feature4', 'feature5']].values  # No reshape needed, it's already 2D
iso_forest.fit(df_reshaped)  # Train the model on the entire dataset

# Define pandas_udf for applying the trained Isolation Forest model
@pandas_udf(IntegerType())
def isolation_forest_udf(pdf: pd.DataFrame) -> pd.Series:
    # The input pdf should be a DataFrame with the correct features (not just one column)
    pdf_reshaped = pdf.values  # No reshape needed for prediction, as it should be 2D
    # Predict using the pre-trained Isolation Forest model
    predictions = iso_forest.predict(pdf_reshaped)
    return pd.Series(predictions)

# Apply the trained model to the DataFrame for predictions
# Use df.select() to get the required columns as a PySpark DataFrame
#df_with_outliers = df.withColumn('outlier', isolation_forest_udf(df.select('feature1', 'feature2', 'feature3', 'feature4', 'feature5')))

#isolation_forest_udf(df_pd[['feature1', 'feature2', 'feature3', 'feature4', 'feature5']])

df_with_outliers = df.withColumn('outlier', iso_forest.predict(df_reshaped))


# Show the results
df_with_outliers.show()

