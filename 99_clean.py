# Databricks notebook source
# MAGIC %sql
# MAGIC DROP DATABASE ml_silver CASCADE;
# MAGIC DROP DATABASE ml_gold CASCADE;
# MAGIC DROP DATABASE feature_store CASCADE;

# COMMAND ----------

import mlflow
from databricks import feature_store

# COMMAND ----------

try:
    experiment_id = mlflow.get_experiment_by_name("/Shared/housing").experiment_id
    mlflow.delete_experiment(experiment_id)
except:
    None

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
client.delete_registered_model(name="housing")

# COMMAND ----------

fs = feature_store.FeatureStoreClient()
#DELETE FEATURE TABLES
