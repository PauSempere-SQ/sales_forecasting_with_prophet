#%% 
import wget 
import requests
import pandas as pd
import numpy as np
import os
import csv
import datetime
import glob

#%%
#get our data 
def get_data():
    years = ['2017', '2018', '2019']
    #years = ['2019']
    months = range(1, 13)

    for y in years: 
        for m in months: 
            if len(str(m)) == 1: 
                _m = '0' + str(m)
            else: 
                _m = str(m)
            
            #get the data from the S3 bucket
            url = 'https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_' + str(y) + '-' + _m + '.csv'
            print('Processing ' + url)

            print('Downloading ...')
            drop_list = [
                'VendorID'
                ,'passenger_count'
                ,'RatecodeID'
                ,'trip_distance'
                ,'tpep_dropoff_datetime'
                ,'store_and_fwd_flag'
                ,'DOLocationID'
                ,'fare_amount'
                ,'extra'
                ,'mta_tax'
                ,'tip_amount'
                ,'tolls_amount'
                ,'improvement_surcharge'
            ]
            df = pd.read_csv(url, parse_dates=True).drop(drop_list, axis = 1)
            df['pickup_year'] = pd.to_datetime(df['tpep_pickup_datetime']).dt.year
            df['pickup_month'] = pd.to_datetime(df['tpep_pickup_datetime']).dt.month
            df['pickup_day'] = pd.to_datetime(df['tpep_pickup_datetime']).dt.day
            df.drop(['tpep_pickup_datetime'], axis = 1)

            print('Casting types and grouping...')

            #group by analysis index 
            df_grouped = df.groupby(by=['pickup_year'
                                        ,'pickup_month'
                                        ,'pickup_day'
                                        ,'PULocationID'
                                        ,'payment_type'], as_index=False)['total_amount'].sum().fillna(0)

            print('Processing file in Parquet form...')
            file_name_parquet = 'D:\\Data\\TaxiData\\yellow_tripdata_' + str(y) + '_' + _m + '.csv'
            df_grouped.to_csv(file_name_parquet, index = False)
            
            print('Done!')

def load_data(): 
    #read all csv files
    full_df = pd.concat([pd.read_csv(f) for f in glob.glob('D:\\Data\\TaxiData\\yellow_tripdata_*.csv')], ignore_index = True)

    full_df.columns = ['year', 'month', 'day', 'location', 'payment_type', 'total_amount']
    full_df['date'] = pd.to_datetime(full_df[['year', 'month', 'day']], yearfirst=True)
    
    #clean some data 
    full_df = full_df.loc[(full_df['year'] >= 2017)  & (full_df['year'] <= 2019)]
    full_df = full_df.drop(['year', 'month', 'day'], axis = 1)

    #split it by time
    df_train = full_df.loc[(full_df['date'] < '2019-06-01')]
    df_test = full_df.loc[(full_df['date'] >= '2019-06-01')]

    df_train = df_train.groupby(['date', 'payment_type', 'location'], as_index=False)['total_amount'].sum()
    df_test = df_test.groupby(['date', 'payment_type', 'location'], as_index=False)['total_amount'].sum()

    return df_train, df_test

# %%
