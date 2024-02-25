import streamlit as st
from datetime import datetime,timedelta
import geopandas as gpd
import folium
from streamlit_folium import folium_static
import hopsworks
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error

from src.config import *
from src.data import *
from src.paths import *

from warnings import simplefilter,filterwarnings
from sklearn.exceptions import InconsistentVersionWarning

#Can ignore this
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

#Investigate into this before supress
filterwarnings(action='ignore', category=InconsistentVersionWarning)

###### Streamlit frontend webapp

# Steamlit styling

st.set_page_config(
   page_title="NYC Taxi demand predictor",
   page_icon=':taxi:',
   layout="wide"
)

st.title("Taxi demand prediction :taxi:")
time_nyc = datetime.now()- timedelta(hours=6)
t = time_nyc.strftime("%d/%m/%y %H:%M") + " ET"
st.caption(f"New York City, USA  -  :blue[{t}]")


###### Predictions

#Connecting to Hopsworks
hw_project = hopsworks.login(project=HOPSWORKS_PROJECT,api_key_value=HOPSWORKS_API_KEY)
fs = hw_project.get_feature_store()
fv = fs.get_feature_view(name=FEATURE_VIEW_NAME, version=FEATURE_VIEW_VERSION)

#Get Test data i.e., data of last 4 months (16 weeks)
fetch_data_from = datetime.now().replace(minute=0,second=0,microsecond=0) - timedelta(weeks=16)
fetch_data_to = datetime.now().replace(minute=0,second=0,microsecond=0) - timedelta(hours=1)
taxi_test_data_ts = fv.get_batch_data(start_time=fetch_data_from,end_time=fetch_data_to)

taxi_test_data_ts.sort_values(by=['pickup_hour', 'pickup_location_id'],inplace=True)
taxi_test_data_ts.columns = ['pickup_time','pickup_location','count_pickup_loc']
taxi_test_data_ts.reset_index(drop=True,inplace=True)
taxi_test_data_ts.to_parquet(TRANSFORMED_PATH + "rides.parquet") #compression='snappy', index=None   

#Transform Time Series data into Tabular Data (Features, Target)
window_size = 672 #1 month i.e., 28 days => 28*24 hours = 672
step_size = 23
features,target = transform_timeseriesdata_into_features_target(window_size,step_size)
print("Features : ",features.shape,"Target : ",target.shape)

X_test = features
y_test = target
X_test['pickup_hour'] = pd.to_datetime(X_test['pickup_hour']).dt.tz_convert(None)

df_test = X_test
df_test['target_rides_next_hour'] = y_test

idx = df_test.groupby('pickup_location_id')['pickup_hour'].idxmax()
df_test_final = df_test.loc[idx]
loc_df = pd.DataFrame({'pickup_location_id': range(1, 266)})
df_test_final = loc_df.merge(df_test_final, how='left', on='pickup_location_id').fillna({'pickup_hour': df_test['pickup_hour'].max()})
df_test_final.fillna(0, inplace=True)

X_pred = df_test_final.drop('target_rides_next_hour',axis=1)
y_actual = df_test_final['target_rides_next_hour'].to_frame()

#Use model from Model Registry in Hopsworks for predictions
hw_project = hopsworks.login(project=HOPSWORKS_PROJECT,api_key_value=HOPSWORKS_API_KEY)
mr = hw_project.get_model_registry()
lgb_model_hw = mr.get_model(name=MODEL_NAME,version=MODEL_VERSION)
lgb_model_hw_path = lgb_model_hw.download()
lgb_model = joblib.load(Path(lgb_model_hw_path)/'nyc_taxi_pipe_model.pkl')
y_pred_lgb = lgb_model.predict(X_pred)

#Prettify the result
y_pred = pd.DataFrame(y_pred_lgb.round(decimals=0).astype(int),columns=['predicted_demand_rides'])
y_pred[['pickup_location_id','pickup_hour']] = X_pred[['pickup_location_id','pickup_hour']]
y_pred = y_pred[['pickup_location_id','pickup_hour','predicted_demand_rides']]
        
# Streamlit display map

txt = y_pred['pickup_hour'].max().strftime("%d/%m/%y %H:%M") + " ET"
st.write(f"Latest data is not available in the Data Warehouse. Predicted demand is for :blue[{txt}]")

st1, st2 = st.columns(2)

nyc_taxi_zones = gpd.read_file(TRANSFORMED_PATH + "taxi_zones")
result_df = nyc_taxi_zones.merge(y_pred, how='left', left_on='LocationID', right_on='pickup_location_id')
top5_pred = result_df.sort_values(by=['predicted_demand_rides'],ascending=False)[:5][['LocationID','borough','zone','predicted_demand_rides']]
top5_pred.columns = ['LocationID','Borough','Zone','Predicted Demand (Rides)']
top5_pred.reset_index(drop=True,inplace=True)
result_df = result_df.drop(['pickup_location_id','pickup_hour'],axis=1)

    
m = folium.Map(location=[40.7128, -74.0060], zoom_start=9)  # Specify the center of NYC
tooltip = folium.GeoJsonTooltip(fields=["LocationID", "borough", "zone","predicted_demand_rides"],aliases=["Location ID: ", "Zone:  ", "Borough: ","Prediced Demand (Rides): "])

c = folium.Choropleth(
    geo_data=result_df,
    data=result_df,
    columns=['LocationID','predicted_demand_rides'],
    key_on='feature.properties.LocationID',
    fill_color='BuPu', 
    fill_opacity=0.7,
    line_opacity=0.2,
    highlight=True,
).add_to(m)

tooltip.add_to(c.geojson)
  
# Trick to hide legend
for key in c._children:
    if key.startswith('color_map'):
        del(c._children[key])
        
#folium.GeoJson(result_df,tooltip=tooltip).add_to(m)

with st1:
    folium_static(m,width=500,height=300)

st2.write("Top 5 locations in NYC with the highest demand")
st2.dataframe(top5_pred,hide_index=True)
