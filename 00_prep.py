# Databricks notebook source
import pandas as pd
import numpy as np

# COMMAND ----------

# MAGIC %md # GET DATA

# COMMAND ----------

from sklearn.datasets import fetch_california_housing

cal_housing = fetch_california_housing(as_frame=True)

# COMMAND ----------

# MAGIC %md ## SPLIT DATA INTO DIMENSION AND FACT

# COMMAND ----------

# MAGIC %md ### House Dimension

# COMMAND ----------

house_dimension = cal_housing['data']

#Add Index
house_index = house_dimension.index.values
house_dimension['HouseID'] = house_index

#Reorder Columns
columns = ['HouseID', 'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
house_dimension = house_dimension[columns]

house_dimension

# COMMAND ----------

# MAGIC %md ### Date Dimension

# COMMAND ----------

dates_dimension = pd.DataFrame({'Date' : pd.date_range('2021-01-01','2021-12-31') })
dates_dimension['Year'] = [x.year for x in dates_dimension['Date']]
dates_dimension['Month'] = [x.month for x in dates_dimension['Date']]
dates_dimension['Day'] = [x.day for x in dates_dimension['Date']]
dates_dimension['DayOfWeek'] = [x.dayofweek for x in dates_dimension['Date']]
dates_dimension['DayOfYear'] = [x.dayofyear for x in dates_dimension['Date']]
dates_dimension

# COMMAND ----------

# MAGIC %md ### Vacation Dimension

# COMMAND ----------

#Generate Random Sell Dates
def random_dates(start, end, n=10):

    start_u = start.value//10**9
    end_u = end.value//10**9

    datetimes = pd.to_datetime(np.random.randint(start_u, end_u, n), unit='s') 
    dates = [x.date() for x in datetimes]
    return dates

start_date = pd.to_datetime('2021-01-01')
end_date   = pd.to_datetime('2021-12-31')

vacation_dates = random_dates(start_date, end_date, 15)
vacation_dimension = pd.DataFrame({
    'Date': vacation_dates,
    'IsVacation': True
})
vacation_dimension

# COMMAND ----------

# MAGIC %md ### Sell Facts

# COMMAND ----------

sell_dates = random_dates(start_date, end_date, house_index.shape[0])

sell_fact = pd.DataFrame({
    'HouseID': house_index,
    'Date': sell_dates,
    'Price': cal_housing['target']})

sell_fact

# COMMAND ----------

# MAGIC %md # SAVE IN METASTORE

# COMMAND ----------

# MAGIC %md ## CREATE DATABASE

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE ml_silver

# COMMAND ----------

# MAGIC %md ## CREATE DIMENSION TABLES

# COMMAND ----------

# MAGIC %md Market Data

# COMMAND ----------

# MAGIC %md House Data

# COMMAND ----------

house = spark.createDataFrame(house_dimension)
house.write.format("delta").mode("overwrite").saveAsTable("ml_silver.house_data")

# COMMAND ----------

# MAGIC %md Dates

# COMMAND ----------

dates = spark.createDataFrame(dates_dimension)
dates.write.format("delta").mode("overwrite").saveAsTable("ml_silver.dates")

# COMMAND ----------

# MAGIC %md Vacation

# COMMAND ----------

vacation = spark.createDataFrame(vacation_dimension)
vacation.write.format("delta").mode("overwrite").saveAsTable("ml_silver.vacations")

# COMMAND ----------

# MAGIC %md ## CREATE FACTS TABLE

# COMMAND ----------

sells = spark.createDataFrame(sell_fact)
sells.write.format("delta").mode("overwrite").saveAsTable("ml_silver.house_sells")

# COMMAND ----------

# MAGIC %md # CHECK

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   h.*
# MAGIC   ,s.Date
# MAGIC   ,d.Year
# MAGIC   ,d.Month
# MAGIC   ,d.Day
# MAGIC   ,d.DayOfWeek
# MAGIC   ,d.DayOfYear
# MAGIC   ,COALESCE(v.IsVacation, False)
# MAGIC   ,s.Price
# MAGIC FROM ml_silver.house_data h
# MAGIC LEFT JOIN ml_silver.house_sells s
# MAGIC ON 1=1
# MAGIC   AND h.HouseID = s.HouseID
# MAGIC LEFT JOIN ml_silver.dates d
# MAGIC ON 1=1
# MAGIC   AND s.Date = d.Date
# MAGIC LEFT JOIN ml_silver.vacations v
# MAGIC ON 1=1
# MAGIC   AND s.Date = v.Date
