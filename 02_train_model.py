# Databricks notebook source
dbutils.widgets.text("start_date", "")
dbutils.widgets.text("end_date", "")

start_date = dbutils.widgets.get("start_date")
end_date = dbutils.widgets.get("end_date")

# COMMAND ----------

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import mlflow
import mlflow.keras
import mlflow.tensorflow

from databricks.feature_store import FeatureStoreClient
from databricks.feature_store import FeatureLookup
import mlflow

# COMMAND ----------

# MAGIC %md # Get House Sell Info

# COMMAND ----------

sells = spark.sql("""
SELECT *
FROM ml_silver.house_sells
WHERE 1=1
    AND Date >= '2021-01-06'
    AND Date <= '2021-06-30'
""")
display(sells)

# COMMAND ----------

# MAGIC %md # Create Training Set

# COMMAND ----------

fs = FeatureStoreClient()

# COMMAND ----------

# MAGIC %md ## Feature Lookups

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES IN feature_store

# COMMAND ----------

# MAGIC %md ### House Features

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM feature_store.house_features

# COMMAND ----------

house_feature_table = "feature_store.house_features"
house_feature_lookups = [
    FeatureLookup( 
        table_name = house_feature_table,
        feature_names = ["MedInc", "AveRooms", "AveBedrms", "Population", "AveOccup"],
        lookup_key = ["HouseID"]
    )
]

# COMMAND ----------

# MAGIC %md ### Date Features

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM feature_store.date_features

# COMMAND ----------

date_feature_table = "feature_store.date_features"
date_feature_lookups = [
    FeatureLookup( 
        table_name = date_feature_table,
        feature_names = ["Month", "DayOfYear"],
        lookup_key = ["Date"]
    )
]

# COMMAND ----------

# MAGIC %md ## Create Training Set

# COMMAND ----------

training_set = fs.create_training_set(
    sells,
    feature_lookups = house_feature_lookups + date_feature_lookups,
    label = "Price",
    exclude_columns=["HouseID", "Date"]
)

training_df = training_set.load_df()
display(training_df)

# COMMAND ----------

# MAGIC %md # Split Training Set

# COMMAND ----------

from sklearn.model_selection import train_test_split

training_pdf = training_df.toPandas()
X_train, X_test, y_train, y_test = train_test_split(training_pdf.drop(columns="Price"),
                                                    training_pdf["Price"],
                                                    test_size=0.2)

# COMMAND ----------

from sklearn.preprocessing import StandardScaler
 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# COMMAND ----------

# MAGIC %md # Train Model

# COMMAND ----------

# MAGIC %md ## Without MLFLOW

# COMMAND ----------

def create_model():
    model = Sequential()
    model.add(Dense(20, input_dim=7, activation="relu"))
    model.add(Dense(20, activation="relu"))
    model.add(Dense(1, activation="linear"))
    return model

# COMMAND ----------

model = create_model()
 
model.compile(loss="mse",
              optimizer="Adam",
              metrics=["mse"])

# COMMAND ----------

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# In the following lines, replace <username> with your username.
experiment_log_dir = "/dbfs/daniel.perez/tb"
checkpoint_path = "/dbfs/daniel.perez>/keras_checkpoint_weights.ckpt"

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=experiment_log_dir)
model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor="loss", mode="min", patience=3)

history = model.fit(X_train, y_train, validation_split=.2, epochs=35, callbacks=[tensorboard_callback, model_checkpoint, early_stopping])

# COMMAND ----------

# MAGIC %load_ext tensorboard
# MAGIC %tensorboard --logdir $experiment_log_dir

# COMMAND ----------

model.evaluate(X_test, y_test)

# COMMAND ----------

import matplotlib.pyplot as plt

keras_pred = model.predict(X_test)
plt.plot(y_test, keras_pred, "o", markersize=2)
plt.xlabel("observed value")
plt.ylabel("predicted value")
plt.axline((0, 0), (1, 1))

# COMMAND ----------

# MAGIC %md ## With MLFLOW

# COMMAND ----------

mlflow.create_experiment("house_prices")

# COMMAND ----------

mlflow.tensorflow.autolog()
 
with mlflow.start_run() as run:
    history = model.fit(X_train, y_train, epochs=35, callbacks=[early_stopping])

    # Save the run information to register the model later
    kerasURI = run.info.artifact_uri

    # Evaluate model on test dataset and log result
    mlflow.log_param("eval_result", model.evaluate(X_test, y_test)[0])

    # Plot predicted vs known values for a quick visual check of the model and log the plot as an artifact
    keras_pred = model.predict(X_test)
    plt.plot(y_test, keras_pred, "o", markersize=2)
    plt.xlabel("observed value")
    plt.ylabel("predicted value")
    plt.savefig("kplot.png")
    mlflow.log_artifact("kplot.png") 