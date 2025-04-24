import joblib
import dagshub
import mlflow
import pandas as pd
import streamlit as st
from pathlib import Path
import datetime as dt
from sklearn.pipeline import Pipeline
from sklearn import set_config
from time import sleep
import dagshub

set_config(transform_output="pandas")

dagshub.init(repo_owner='Ubaidmalik9567', repo_name='Uber-Demand-Prediction', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Ubaidmalik9567/Uber-Demand-Prediction.mlflow")

# get model name
registered_model_name = 'uber_demand_prediction_model'
stage = "Production"
model_path = f"models:/{registered_model_name}/{stage}"

# load the latest model from model registry
model = mlflow.sklearn.load_model(model_path)

# set the root path
root_path = Path(__file__).parent
# path of the data
plot_data_path = root_path / "data/external/plot_data.csv"
data_path = root_path / "data/processed/test.csv"

# model paths
kmeans_path = root_path / "models/mb_kmeans.joblib"
scaler_path = root_path / "models/scaler.joblib"
encoder_path = root_path / "models/encoder.joblib"
model_path = root_path / "models/model.joblib"

# load the objects
scaler = joblib.load(scaler_path)
encoder = joblib.load(encoder_path)
model = joblib.load(model_path)
kmeans = joblib.load(kmeans_path)

# dataset to plot
df_plot = pd.read_csv(plot_data_path)
df = pd.read_csv(data_path, parse_dates=["tpep_pickup_datetime"]).set_index("tpep_pickup_datetime")

# UI of app
# make the title for the page
st.title("Uber Demand in New York City 🚕🌆")

# select for only neighbors or all
st.sidebar.title("Options")
map_type = st.sidebar.radio(label="Select the type of Map",
                     options=["Complete NYC Map", "Only for Neighborhood Regions"],
                     index=1)

# select the date
st.subheader("Date")
date = st.date_input("Select the date", value=None,
                     min_value=dt.date(year=2016, month=3, day=1),
                     max_value=dt.date(year=2016, month=3, day=31)) 
st.write("**Date:**", date)

# select the time of day
st.subheader("Time")
time = st.time_input("Select the time", value=None)
st.write("**Current Time:**", time)

if date and time:

       # next time interval
       delta = dt.timedelta(minutes=15)
       next_interval = dt.datetime(year=date.year,
                                   month=date.month,
                                   day=date.day, 
                                   hour=time.hour, 
                                   minute=time.minute) + delta
       st.write("Demand for Time: ", next_interval.time())

       # create the datetime index
       index = pd.Timestamp(f"{date} {next_interval.time()}")
       st.write("**Date & Time:**", index)
       
       # sample a latitude longitude value
       st.subheader("Location")
       sample_loc = df_plot.sample(1).reset_index(drop=True)
       lat = sample_loc["pickup_latitude"].item()
       long = sample_loc["pickup_longitude"].item()
       region = sample_loc["region"].item()
       st.write("**Your Current Location**")
       st.write(f"Lat: {lat}")
       st.write(f"Long: {long}")
       
       with st.spinner("Fetching your Current Region"):
              sleep(3)
              
       st.write("Region ID: ", region)
       # scale the data
       scaled_cord = scaler.transform(sample_loc.iloc[:, 0:2])
       
       # plot the map
       st.subheader("MAP")
       
       # list of 30 hex colors on a white background with 8 digits
       colors = ["#FF0000", "#FF4500", "#FF8C00", "#FFD700", "#ADFF2F", 
              "#32CD32", "#008000", "#006400", "#00FF00", "#7CFC00", 
              "#00FA9A", "#00FFFF", "#40E0D0", "#4682B4", "#1E90FF", 
              "#0000FF", "#0000CD", "#8A2BE2", "#9932CC", "#BA55D3", 
              "#FF00FF", "#FF1493", "#C71585", "#FF4500", "#FF6347", 
              "#FFA07A", "#FFDAB9", "#FFE4B5", "#F5DEB3", "#EEE8AA"]

       # add color to the data
       region_colors = {region: colors[i] for i, region in enumerate(df_plot["region"].unique().tolist())}
       df_plot["color"] = df_plot["region"].map(region_colors)
       
       # make prediction pipeline
       pipe = Pipeline([
              ('encoder',encoder),
              ('reg',model)
       ])
       

