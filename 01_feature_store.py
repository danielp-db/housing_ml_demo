# Databricks notebook source
from databricks import feature_store
from pyspark.sql.types import DateType
import pyspark.sql.functions as F

# COMMAND ----------

# MAGIC %md ### Configure

# COMMAND ----------

fs = feature_store.FeatureStoreClient()

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS feature_store

# COMMAND ----------

# MAGIC %md # Create Feature Table

# COMMAND ----------

# MAGIC %md ## House Data

# COMMAND ----------

# MAGIC %md ### View Data

# COMMAND ----------

house = spark.sql("SELECT * FROM ml_silver.house_data")
display(house)

# COMMAND ----------

# MAGIC %md ### Transform Data

# COMMAND ----------

def transform_house(df):
    return (
        df
        .withColumn("HouseAge", F.expr("""case
            when HouseAge >= 40 then '40+'
            when HouseAge >= 20 then '20-40'
            when HouseAge >= 5 then '5-20'
            when HouseAge >= 0 then '0-5'
            else 'Unknown'
            end"""))
        .withColumn("Location", F.when(F.col("Latitude").between(37,38), "Downtown").otherwise("Metro"))
        .select(['HouseID', 'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Location'])
    )

# COMMAND ----------

house_features = transform_house(house)
display(house_features)

# COMMAND ----------

# MAGIC %md ### Create Feature Table

# COMMAND ----------

fs.create_table(
    name="feature_store.house_features",
    primary_keys=["HouseID"],
    df=house_features,
    #partition_columns="yyyy_mm",
    description="House Features",
)

# COMMAND ----------

# MAGIC %md ## Dates

# COMMAND ----------

# MAGIC %md ### View Data

# COMMAND ----------

dates = spark.sql("SELECT * FROM ml_silver.dates")
display(dates)

# COMMAND ----------

vacations = spark.sql("SELECT * FROM ml_silver.vacations")
display(vacations)

# COMMAND ----------

# MAGIC %md ### Transform Data

# COMMAND ----------

def transform_dates(dates, vacations):
    transformed_dates = (
        dates
        .join(vacations, on="Date", how="left")
        .withColumn("IsVacation", F.coalesce(F.col("IsVacation"), F.lit(False)))
        .withColumn("Date", F.col("Date").astype(DateType()))
    )
    return transformed_dates

# COMMAND ----------

date_features = transform_dates(dates, vacations)
display(date_features)

# COMMAND ----------

# MAGIC %md ### Create Feature Tables

# COMMAND ----------

fs.create_table(
    name="feature_store.date_features",
    primary_keys=["Date"],
    df=date_features,
    #partition_columns="yyyy_mm",
    description="Date Features"
)