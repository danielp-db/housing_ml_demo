# Databricks notebook source
dbutils.widgets.text("01_start_date", "2021-12-01")
dbutils.widgets.text("02_end_date", "2021-12-31")
dbutils.widgets.dropdown("03_model_version", "None", choices=["None","Staging","Production"])

start_date = dbutils.widgets.get("01_start_date")
end_date = dbutils.widgets.get("02_end_date")
model_version = dbutils.widgets.get("03_model_version")

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType
import pandas as pd

# COMMAND ----------

import mlflow
from databricks import feature_store

# COMMAND ----------

fs = feature_store.FeatureStoreClient()

# COMMAND ----------

# MAGIC %md # GET DATA

# COMMAND ----------

sells = spark.sql(f"""
SELECT *
FROM ml_silver.house_sells
WHERE 1=1
    AND Date >= '{start_date}'
    AND Date <= '{end_date}'
""")
display(sells)

# COMMAND ----------

house_feature_table = "feature_store.house_features"
house_feature_lookups = [
    feature_store.FeatureLookup( 
        table_name = house_feature_table,
        feature_names = ["MedInc", "AveRooms", "AveBedrms", "Population", "AveOccup"],
        lookup_key = ["HouseID"]
    )
]

# COMMAND ----------

date_feature_table = "feature_store.date_features"
date_feature_lookups = [
    feature_store.FeatureLookup( 
        table_name = date_feature_table,
        feature_names = ["Month", "DayOfYear"],
        lookup_key = ["Date"]
    )
]

# COMMAND ----------

scoring_set = fs.create_training_set(
    sells,
    feature_lookups = house_feature_lookups + date_feature_lookups,
    label = "Price",
    #exclude_columns=["HouseID", "Date"]
)

scoring_df = scoring_set.load_df()
display(scoring_df)

# COMMAND ----------

# MAGIC %md # GET MODEL

# COMMAND ----------

model = mlflow.pyfunc.load_model(f"models:/housing/{model_version}")
model_udf = mlflow.pyfunc.spark_udf(spark, f"models:/housing/{model_version}")

# COMMAND ----------

# MAGIC %md ## Scoring Parallel

# COMMAND ----------


@F.pandas_udf(returnType=DoubleType())
def predict_pandas_udf(*cols):
    # cols will be a tuple of pandas.Series here.
    X = pd.concat(cols, axis=1).values
    return pd.Series(model.predict(X).reshape(-1))

independent_vars = ['MedInc', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Month','DayOfYear']
housing_preds = scoring_df.withColumn("PredictedPrice", predict_pandas_udf(*independent_vars))
display(housing_preds)

# COMMAND ----------

# MAGIC %md SAVE DATA

# COMMAND ----------

housing_preds.write.format("delta").mode("overwrite").saveAsTable("ml_gold.housing_preds")

# COMMAND ----------

# MAGIC %md ## Scoring SingleNode

# COMMAND ----------

scoring_pdf = scoring_df.toPandas()[['MedInc', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Month','DayOfYear']].values
model.predict(scoring_pdf)
