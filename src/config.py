from dotenv import load_dotenv
import os
from src.paths import *

load_dotenv(ENV_PATH+'/.env')

HOPSWORKS_PROJECT = 'taxi_demand_prediction'
HOPSWORKS_API_KEY = os.environ['HOPSWORKS_API_KEY']

FEATURE_GROUP_NAME = 'taxi_time_series_hourly'
FEATURE_GROUP_VERSION = 1

FEATURE_VIEW_NAME = 'taxi_time_series_hourly_feature_view'
FEATURE_VIEW_VERSION = 1

MODEL_NAME = 'taxi_demand_predictor_next_hour'
MODEL_VERSION = 1