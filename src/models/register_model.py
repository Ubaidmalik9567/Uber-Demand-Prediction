import json
import mlflow
import dagshub
import logging
from pathlib import Path
from mlflow.client import MlflowClient
import dagshub

dagshub.init(repo_owner='Ubaidmalik9567', repo_name='Uber-Demand-Prediction', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Ubaidmalik9567/Uber-Demand-Prediction.mlflow")

# create a logger
logger = logging.getLogger("register_model")
logger.setLevel(logging.INFO)

# attach a console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

# make a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
