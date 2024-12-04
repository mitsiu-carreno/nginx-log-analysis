# Generalities
This repo stores all code and setup required to perform an analysis of logs generated by nginx. This analysis is performed using spark framework in several containers and minio to provide storage.

## Run spark and minio containers
```bash
podman build -t cluster-apache-spark:3.5.3 .
```

```bash
podman-compose up
```

## Run code
Code is stored in apps folder and can be run with
```bash
podman exec -it spark-master ./bin/spark-submit --packages org.apache.hadoop:hadoop-aws:3.3.4 /opt/spark-apps/script.py
```

While running on a beefier computer I used the following command:
```bash
docker exec -it spark-master ./bin/spark-submit \
--packages org.apache.hadoop:hadoop-aws:3.3.4 \
--conf spark.executor.memory=50g \
--conf spark.executor.memoryOverhead=1000m \
--executor-memory 50g \
--driver-memory 50g \
/opt/spark-apps/file.py 2>&1 | tee logs/0.log
```

```
docker exec -it spark-master ./bin/spark-submit --packages org.apache.hadoop:hadoop-aws:3.3.4 --conf spark.executor.memory=100g --conf spark.executor.memoryOverhead=10000m --executor-memory 100g --driver-memory 100g /opt/spark-apps/1-extract.py 2>&1 | tee logs/1.log && docker exec -it spark-master ./bin/spark-submit --packages org.apache.hadoop:hadoop-aws:3.3.4 --conf spark.executor.memory=100g --conf spark.executor.memoryOverhead=10000m --executor-memory 100g --driver-memory 100g /opt/spark-apps/2-train_domain_classifier.py 2>&1 | tee logs/2.log && docker exec -it spark-master ./bin/spark-submit --packages org.apache.hadoop:hadoop-aws:3.3.4 --conf spark.executor.memory=100g --conf spark.executor.memoryOverhead=10000m --executor-memory 100g --driver-memory 100g /opt/spark-apps/3-predict_domain.py 2>&1 | tee logs/3.log
```

### Scala
https://spark.apache.org/docs/latest/quick-start.html#self-contained-applications
Structure
```bash
tree .
.
├── build.sbt
└── src
    └── main
        └── scala
            └── App.scala
```

```bash
sbt package
```

```bash
podman exec -it spark-master ./bin/spark-submit --class "ScalaApp" --packages org.apache.hadoop:hadoop-aws:3.3.4 /opt/spark-apps/98-sbt-template/target/scala-2.12/anomaly-detection_2.12-1.0.jar
```

```bash
podman exec -it spark-master ./bin/spark-shell  --packages org.apache.hadoop:hadoop-aws:3.3.4,com.linkedin.isolation-forest:isolation-forest_3.5.0_2.12:3.0.6,org.elasticsearch:elasticsearch-spark-30_2.12:8.6.2 --conf spark.hadoop.fs.s3a.endpoint=http://minio:9000 --conf spark.hadoop.fs.s3a.access.key=accesskey --conf spark.hadoop.fs.s3a.secret.key=secretkey --conf spark.hadoop.fs.s3a.path.style.access=true --conf spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem --conf spark.es.nodes="es01" --conf spark.es.port="9200" --conf spark.es.nodes.wan.only="true"
```

To update the code in the beefier computer:
```bash
scp apps/*  designar4@192.168.100.20:/home/designar4/spark-cluster/compose/apps/

ssh -L :12345:localhost:4040 designar4@192.168.100.20
```

### Screen
Some commands are to long to keep a ssh connection open, that's why I used screen
```bash
# Start new named screen session
screen -S spark

# Detach from named session
Ctrl + A, then D

# Can even break the ssh connection

# List sessions
screen -ls

# Reattach to session
screen -r spark
```

# Code formatting
```bash
#Formatting
black file.py 

#Lintting
flake8 file.py
pylint file.py
```

# Minio mc
```bash
mc alias set myminio http://minio:9000 accesskey secretkey
mc cp -r myminio/logs/output/extract/domain_category=unknown myminio/logs/output/extract/domain_category=other
mc du -r myminio/logs/output/
```

# Usefull spark code snippets:
```python
row = df.take(1)
print(row)
```

```python
print(f"{df.count()}, {len(df.columns)}")

df.printSchema()
```

```python
rows = df.limit(10020).collect()[10000:10020]
df_subset = spark.createDataFrame(rows, df.schema)
df_subset.show(truncate=False)
```

```python
df_exploded = df_exploded
    .withColumn("query_key", split(col("query_param"), "=").getItem(0))
    .withColumn("query_value", split(col("query_param"), "=").getItem(1))
```

```python
df.filter(
    (col("domain") != "domain1") 
    & (df["query_param"].isNotNull())
).groupBy("domain").count().show(truncate=False)
```

```scala
import org.apache.spark.sql.types._

val df2 = df.withColumn("anomaly_bool", col("anomaly").cast(BooleanType))

df2.select("fabstime", "status", "body_bytes_sent", "outlierScore", "anomaly_bool").write.format("es").mode("overwrite").save("anom")
```

```python
```

```python
```

```python
```

ToDo:
- [ ] Domain and method encoder from all data
- [ ] Confussion matrix in regression
- [ ] Anomally execution
- [ ] Anomally viz  
