#%%
import pandas as pd
import numpy as np 
import fbprophet as prop
import functions
import matplotlib.pyplot as plt 
import os
import csv
import glob
import datetime
import joblib
import time 

#%%
#get our data
df_train, df_test = functions.load_data()

#%%
#create the group by object 
groups = df_train.groupby(['location', 'payment_type'])
forecasts = pd.DataFrame()

print('processing ' + str(len(groups)) + ' partitions ')

#%%
s = time.time()

for group_name, g in groups: 

    location = str(g['location'].unique())
    payment_type = str(g['payment_type'].unique())

    #ensure that we have exactly one day per partition and we fit appropiate column names
    partition = g.groupby(['date'], as_index=False)['total_amount'].sum()
    partition.columns = ['ds', 'y']

    #minimum 60 data points
    if partition.shape[0] > 60: 
        print('processing partition. Location: ' + str(group_name[0]) + ' Payment type: ' + str(group_name[1]))
    
        #train on partition 
        model = prop.Prophet()
        model.fit(partition)

        # #test it with half a year 
        future = model.make_future_dataframe(periods=210)
        forecast = model.predict(future)

        forecast['location'] = location
        forecast['payment_type'] = payment_type

        #accumulate in forecasts
        #it's not the fastest way in the world, probably with ndarrays would be faster but it's easier from a developers point of view
        forecasts = forecasts.append(forecast)

e = time.time() 

#%%
print(e-s)
#3k seconds --> 50 minutes

#%%
#check the output format
forecast.columns

#%%
#save forecasts and actual data 
cols_to_drop = [
    'trend', 
    'additive_terms', 
    'additive_terms_lower', 'additive_terms_upper',
    'weekly', 'weekly_lower', 'weekly_upper', 'yearly', 'yearly_lower',
    'yearly_upper', 'multiplicative_terms', 'multiplicative_terms_lower',
    'multiplicative_terms_upper'
]

df_test.to_csv('D:\\Data\\TaxiData\\test.csv', index = False)
forecasts.drop(cols_to_drop, axis = 1).to_csv('D:\\Data\\TaxiData\\multiple_forecasts.csv', index = False)

#%%
#parallel version
from joblib import Parallel, delayed, parallel 

#create a function to embed the training and 
def train_prophet(g):
    #receives the grouped tubple, split it
    data = g[1]
    name = g[0]

    location = str(data['location'].unique())
    payment_type = str(data['payment_type'].unique())

    #ensure that we have exactly one day per partition 
    partition = data.groupby(['date'], as_index=False)['total_amount'].sum()
    partition.columns = ['ds', 'y']

    #minimum 60 data points, around 2 months
    if partition.shape[0] >= 60:
        #train on partition 
        model = prop.Prophet()
        model.fit(partition)

        # #test it with half a year 
        future = model.make_future_dataframe(periods=210)

        forecast = model.predict(future)

        forecast['location'] = str(location)
        forecast['payment_type'] = str(payment_type)

        cols_to_drop = [
            'trend', 
            'additive_terms', 
            'additive_terms_lower', 'additive_terms_upper',
            'weekly', 'weekly_lower', 'weekly_upper', 'yearly', 'yearly_lower',
            'yearly_upper', 'multiplicative_terms', 'multiplicative_terms_lower',
            'multiplicative_terms_upper'
        ]

        file_name = 'D:\\Data\\TaxiData\\forecast_results\\forecasts_' + str(name[0]) + '_' + str(name[1]) + '.csv'

        #save results and return a simplified version of the results
        forecast.drop(cols_to_drop, axis = 1).to_csv(file_name, index = False)
        
        return forecast.drop(cols_to_drop, axis = 1)
    else: 
        return pd.DataFrame()

#%%
#parallel for loop version
s = time.clock()
forecasts = Parallel(n_jobs = 8, verbose = 1)(delayed(train_prophet)(g) for g in groups)
e = time.clock()
e - s
#15 minutes with 8 cores and 8 processes


# %%
