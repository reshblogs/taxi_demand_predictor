# Functions used in Data preprocessing steps

import requests
from bs4 import BeautifulSoup
import re
import os
import pandas as pd
import numpy as np
import plotly.express as px

from src.paths import *

url = 'https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page'


##### Download raw data

def download_files_raw(year):
    
    resp_url = requests.get(url)
    if resp_url.status_code == 200 :
        soup = BeautifulSoup(resp_url.text,"html.parser")
        yellow_a_tags = soup.find_all(title="Yellow Taxi Trip Records",href=re.compile(str(year)))
        files = []
        for link in yellow_a_tags:
            yellow_href = link.get('href')
            #Getting an extra %20 at the end of file name due to space in the original a tag on the website
            yellow_href = yellow_href.strip() 
            resp_file = requests.get(yellow_href)
            file_name = resp_file.url.split('/')[4]
            fn = file_name
            file_name = file_name.replace('yellow_tripdata','rides')
            os.chdir(RAW_PATH)
            with open(file_name,'wb') as f:
                files.append(file_name)
                f.write(resp_file.content)

        print("Downloaded files : ",files)
        
    else :
        return "Website not found"
    

    
##### Validate the raw data

def validate_data_files():
    validated_files = []
    raw_files = [f for f in os.listdir(RAW_PATH) if 'parquet' in f]
    raw_files.sort()
    
    all_df = []
    for f in raw_files :
        #print("### Original file = ",f)
        f_path = RAW_PATH + '/' + f
        f_y_m = f.split('_')[1].split('.')[0].split('-')
        f_y_m = [int(val) for val in f_y_m]
        df = pd.read_parquet(path = f_path)
        #print("Original Data : ",df.shape)
        
        df = df[['tpep_pickup_datetime','PULocationID']]
        df.columns = ['pickup_time','pickup_location']  

        # Retain only rows that have correct year and month as per data file
        df['pickup_year'] = df['pickup_time'].dt.year
        df['pickup_month'] = df['pickup_time'].dt.month
        df = df[df['pickup_year'] == f_y_m[0]]
        df = df[df['pickup_month'] == f_y_m[1]]

        df = df[['pickup_time','pickup_location']]
        all_df.append(df)
        #print("Validated Data : ",df.shape)
        
        #display(df['pickup_time'].describe(datetime_is_numeric=True))
        validated_files.append(f)
        val_path = VALIDATED_PATH + f
        df.to_parquet(path=val_path) #compression='snappy', index=None
        
    #Delete raw files
    for f in raw_files:
        os.remove(RAW_PATH + '/' + f)

    all_df = pd.concat(all_df)
    all_df.reset_index(drop=True,inplace=True) #as index numbers are weird
    return all_df


##### Transform the validated data into Time Series data

def handle_missing_indexes_ts(df):
    
    all_loc = df.pickup_location.unique()
    all_loc.sort()
    #print("Unique no.of locations in dataset = ",len(all_loc))

    df.set_index('pickup_time',inplace=True)  #Make pickup_time as index i.e., DateTimeIndex
    df = df.groupby('pickup_location').resample('1H').count()
    df = df.rename(columns={'pickup_location': 'count_pickup_loc'})
    df.reset_index(inplace=True) #Make all indexes as columns
    df.set_index('pickup_time',inplace=True) #Make pickup_time as index i.e., DateTimeIndex
    df = df.sort_index()
    df.reset_index(inplace=True) #Make all indexes as columns
    #display(df)

    # Create a new dataframe with all the possible times and locations, and assign 0 for missing values
    min_d = df['pickup_time'].min()
    max_d = df['pickup_time'].max()
    dates_idx = pd.date_range(start=min_d,end=max_d,freq='1H')
    col_time = np.repeat(dates_idx,len(all_loc))
    #print(col_time.shape) #1d array

    all_loc_arr = np.array(all_loc).reshape(-1,1)
    col_loc = all_loc_arr
    for i in range(0,len(dates_idx)-1):
        col_loc = np.concatenate((col_loc,all_loc_arr))
    col_loc = col_loc.flatten() #Convert 2d array to 1d array
    #print(col_loc.shape)

    all_df = pd.DataFrame({'pickup_time':col_time,
                            'pickup_location':col_loc,
                            'count_pickup_loc':0})
    all_df

    final_df = pd.concat([df,all_df],ignore_index=True)
    #print(final_df.shape)
    final_df = final_df.drop_duplicates(subset=['pickup_time','pickup_location'],keep='first')
    final_df.reset_index(drop=True,inplace=True)
    final_df

    return final_df


def transform_to_timeseries():
    
    validated_files = [f for f in os.listdir(VALIDATED_PATH) if 'parquet' in f]
    validated_files.sort()
    
    df = []
    
    for f in validated_files:
        f_path = VALIDATED_PATH + f
        one_df = pd.read_parquet(path = f_path)
        df.append(one_df)
    
    df = pd.concat(df)
    df.reset_index(drop=True,inplace=True) #as index numbers are weird
    #print(df.shape)
    
    df = handle_missing_indexes_ts(df)
    
    trans_path = TRANSFORMED_PATH + "rides.parquet"
    df.to_parquet(path=trans_path) #compression='snappy', index=None
    
    #Delete validated files
    for f in validated_files:
        os.remove(VALIDATED_PATH + '/' + f)

    return df
 
##### Transform Time Series data into Training Data (Features, Target)
    
def transform_timeseriesdata_into_features_target(window_size,step_size):
    
    transformed_file = [f for f in os.listdir(TRANSFORMED_PATH) if 'parquet' in f][0]
    transformed_file

    features = []
    target = []
    col_names = [f'rides_previous_{el}_hours' for el in range(window_size,0,-1)]
    
    
    f_path = TRANSFORMED_PATH + transformed_file
    df = pd.read_parquet(path = f_path)
    
    all_loc = df.pickup_location.unique()
    all_loc.sort()
    
    for loc_id in all_loc:
        df_single_loc = df[df.pickup_location == loc_id]
        df_single_loc = df_single_loc[['pickup_time','count_pickup_loc']]
        df_single_loc.columns = ['pickup_hour','rides']

        i_start = 0
        i_end = len(df_single_loc) - window_size

        X = []
        y = []
        y_pickup_hour = []

        for i in range(i_start,i_end, step_size):
            j = i + window_size
            X.append(df_single_loc['rides'][i:j].tolist()) 
            y.append(df_single_loc['rides'].iloc[j])
            y_pickup_hour.append(df_single_loc['pickup_hour'].iloc[j])

        features_single_loc = pd.DataFrame(X,columns = col_names)
        features_single_loc['pickup_hour'] = y_pickup_hour
        features_single_loc['pickup_location_id'] = loc_id
        target_single_loc = pd.DataFrame(y,columns = ['target_rides_next_hour'])

        features.append(features_single_loc)
        target.append(target_single_loc)
        
    features = pd.concat(features)
    features.reset_index(drop=True,inplace=True) #as index numbers are weird

    target = pd.concat(target)
    target.reset_index(drop=True,inplace=True) #as index numbers are weird
    
    # Save features and target data back to disk
    f_path = TRANSFORMED_PATH + "features.parquet"
    features.to_parquet(path=f_path) #compression='snappy', index=None
    t_path = TRANSFORMED_PATH + "target.parquet"
    target.to_parquet(path=t_path) #compression='snappy', index=None
    
    #Delete transformed file
    os.remove(TRANSFORMED_PATH + transformed_file)
        
    return features,target



##### Data Visualization


