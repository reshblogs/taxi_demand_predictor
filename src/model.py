from geopy.geocoders import ArcGIS
from sklearn.preprocessing import FunctionTransformer
import lightgbm as lgb
from sklearn.pipeline import make_pipeline

geolocator_arcgis = ArcGIS()

def geocode_taxi_locations(address):
    if address != "No":
        loc = geolocator_arcgis.geocode(address)
    else:
        return 0.0,0.0
    return loc.latitude,loc.longitude

def avg_rides_last4weeks(X):
    avg_X = (X['rides_previous_168_hours'] + X['rides_previous_336_hours'] + X['rides_previous_504_hours'] + X['rides_previous_672_hours'])/4
    X['avg_rides_last4weeks'] = avg_X
    return X

def extract_features_pickuptime(X):
    pick = X['pickup_hour']
    X['pickup_hours'] = pick.dt.hour
    X['pickup_dayofweek'] = pick.dt.dayofweek
    X['pickup_isyearstart'] = pick.dt.is_year_start
    X['pickup_isyearend'] = pick.dt.is_year_end
    X = X.replace({True:1,False:0})
    X = X.drop('pickup_hour',axis=1)
    return X


def get_pipeline_no_hyperparametertuning():
    
    ### Add new feature by taking an average of rides taken in the last 4 weeks
    transformer_add_feature_avgrides = FunctionTransformer(avg_rides_last4weeks)
    
    ### Extract new features from pickup timestamp
    transformer_extract_features_pickuptime = FunctionTransformer(extract_features_pickuptime)
    
    ### Create LGBM regressor
    lgb_fepipe_reg = lgb.LGBMRegressor()
    
    ### Create feature engineering pipeline
    pipe = make_pipeline(transformer_add_feature_avgrides,
                     transformer_extract_features_pickuptime,
                     lgb_fepipe_reg)
    return pipe


def get_pipeline(regressor_params={}):
    
    ### Add new feature by taking an average of rides taken in the last 4 weeks
    transformer_add_feature_avgrides = FunctionTransformer(avg_rides_last4weeks)
    
    ### Extract new features from pickup timestamp
    transformer_extract_features_pickuptime = FunctionTransformer(extract_features_pickuptime)
    
    ### Create LGBM regressor
    lgb_fepipe_reg = lgb.LGBMRegressor(**regressor_params)
    
    ### Create feature engineering pipeline
    pipe = make_pipeline(transformer_add_feature_avgrides,
                     transformer_extract_features_pickuptime,
                     lgb_fepipe_reg)
    return pipe
  
    

    
    
    