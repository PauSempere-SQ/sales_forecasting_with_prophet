#%%
import pandas as pd
import numpy as np 
import fbprophet as prop
#import download_files as dw 
import matplotlib.pyplot as plt 
import os
import csv
import glob
import datetime

#%%
#download data in case it is not downloaded
#dw.get_data() 

#%%
#read all csv files
full_df = pd.concat([pd.read_csv(f) for f in glob.glob('D:\\Data\\TaxiData\\yellow_tripdata_*.csv')], ignore_index = True)

full_df.columns = ['year', 'month', 'day', 'location', 'payment_type', 'total_amount']
full_df['date'] = pd.to_datetime(full_df[['year', 'month', 'day']], yearfirst=True)
full_df.describe()

#%%
#clean some data 
full_df = full_df.loc[(full_df['year'] >= 2017)  & (full_df['year'] <= 2019)]
full_df.describe()

#%%
full_df = full_df.drop(['year', 'month', 'day'], axis = 1)

#%%
#plot it 
full_df.plot(x='date', y='total_amount', figsize = (15, 15))

#%%
#split it by time
df_train = full_df.loc[(full_df['date'] < '2019-06-01')]
df_test = full_df.loc[(full_df['date'] >= '2019-06-01')]

#%%
df_train = df_train[['date', 'total_amount']]
df_test = df_test[['date', 'total_amount']]
#prophet standard 
df_train.columns = ['ds', 'y']
df_test.columns = ['ds', 'y']

df_train = df_train.groupby('ds', as_index=False)['y'].sum()
df_test = df_test.groupby('ds', as_index=False)['y'].sum()

#%%
df_train.plot(x='ds', y='y', figsize = (15, 15))

#%%
#fit a basic prophet model 
model = prop.Prophet()
model.fit(df_train)

#%%
#test it with half a year 
future = model.make_future_dataframe(periods=210)

forecast = model.predict(future)
fig_forecast = model.plot(forecast)

#%%
#check components
fig_components = model.plot_components(forecast)

#%%
preds = forecast.loc[forecast['ds'] >= '2019-06-01']

fig = plt.figure(figsize=(10, 10))
plt.plot(preds['ds'], preds['yhat'], color = 'blue')
plt.plot(df_test['ds'], df_test['y'], color = 'red')
plt.show()


# %%
