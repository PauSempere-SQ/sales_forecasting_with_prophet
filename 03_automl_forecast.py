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
from azureml.core import Workspace, Datastore, Dataset, ScriptRunConfig, ComputeTarget, Experiment
from azureml.data.datapath import DataPath
from azureml.train.sklearn import SKLearn
from azureml.train.estimator import Estimator
from azureml.train.automl import AutoMLConfig
from azureml.train.automl.run import AutoMLRun
import azureml.dataprep as dprep
from azureml.data.datapath import DataPath
from azureml.dataprep import col
from azureml.core.authentication import InteractiveLoginAuthentication

#multi tenant with my account 
int_auth = InteractiveLoginAuthentication(tenant_id='35069d74-1489-4194-80c7-3a81385ead5b')
ws = Workspace.from_config(auth=int_auth)
print(ws.name)

#%%
#get our data
df_train, df_test = functions.load_data()

#get compute context 
amlcompute_cluster = "aml-automl" 

#LOCATE OUR COMPUTE RESOURCE IN OUR WORKSPACE
aml_cluster = ComputeTarget(workspace=ws, name=amlcompute_cluster)

#%%
ds_train = Dataset.get_by_name(ws, 'sales_train')
ds_test = Dataset.get_by_name(ws, 'sales_test')

ds_train.take(10).to_pandas_dataframe().head(10)

#%%
#configure automl run in AzureML Compute
experiment_name = 'automl-salesdemo-amlcompute'
experiment = Experiment(ws, experiment_name)

#configure the remote execution
automl_config = AutoMLConfig(task = 'forecasting',
                             debug_log = 'automl_errors_regression.log',
                             primary_metric = 'normalized_root_mean_squared_error',
                             iteration_timeout_minutes = 15,
                             experiment_timeout_hours=1,
                             #create N time series grouping by the IDs
                             grain_column_names = ['location', 'payment_type'],
                             time_column_name = 'date',
                             enable_dnn = True,
                             compute_target=aml_cluster,
                             iterations = 20,
                             max_cores_per_iteration = -1, #as many as your maximum worker nodes have
                             max_concurrent_iterations = 4, #change it based on number of worker nodes in an AMLCompute context
                             featurization = 'auto',
                             training_data = ds_train,
                             validation_data = ds_test, #t
                             model_explainability=True,
                             label_column_name = 'total_amount'
                             )


# %%
run = experiment.submit(automl_config)
run.wait_for_completion(show_output = True)

#%%
#review results
best_run, fitted_model = run.get_output()
print(best_run)
print(fitted_model)

#%%
df_test_preds = df_test.copy() 
df_test_preds['yhat'] = fitted_model.predict(df_test)


