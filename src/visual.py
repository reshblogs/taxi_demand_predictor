import pandas as pd
from datetime import timedelta
import plotly.express as px

##### Plot a sample of training data

def plot_training_sample(features,target,row_id):
    
    fe = features.iloc[row_id][:-2]
    pickup_hour = features.iloc[row_id][-2]
    pickup_loc_id = features.iloc[row_id][-1]
    start_date = pickup_hour - timedelta(hours=len(fe))
    fe_times = pd.date_range(start=start_date,end=pickup_hour,freq='1H')

    fig = px.line(x=fe_times[:-1],y=fe,title="Visual plot of rides before pickup hour "+str(pickup_hour)+" at location "+str(pickup_loc_id))
    fig.add_scatter(x=fe_times[-1:],y=target.iloc[row_id],name="Actual value",line_color='brown')
    return fig