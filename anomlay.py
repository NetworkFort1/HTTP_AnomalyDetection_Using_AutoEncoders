#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import glob
import os
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, concatenate
from zat.log_to_dataframe import LogToDataFrame


# In[2]:


# Function to read and process a log file using LogToDataFrame
def read_log_to_dataframe(file_path):
    log_to_df = LogToDataFrame()
    return log_to_df.create_dataframe(file_path)

log_folder = "attacked_data"
output_folder = "all_dataframes_csv"  # Specify the output folder path

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# List to store all dataframes
all_dataframes = []

# Iterate through log files in the folder and create dataframes
for log_file in os.listdir(log_folder):
    if log_file.endswith(".log"):
        full_log_path = os.path.join(log_folder, log_file)
        df = read_log_to_dataframe(full_log_path)
        all_dataframes.append(df)

# Save each dataframe to a CSV file in the output folder
for index, df in enumerate(all_dataframes):
    output_file_path = os.path.join(output_folder, f"dataframe_{index}.csv")
    df.to_csv(output_file_path, index=False)


# In[3]:


input_folder = "all_dataframes_csv"  
output_file = "combined_attack.csv"  

# Get a list of all CSV files in the input folder
csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

# Initialize an empty list to store dataframes
dataframes = []

# Read each CSV file and store its contents as a dataframe
for csv_file in csv_files:
    csv_path = os.path.join(input_folder, csv_file)
    df = pd.read_csv(csv_path)
    dataframes.append(df)

# Concatenate all dataframes into a single dataframe
combined_df = pd.concat(dataframes, ignore_index=True)

# Save the combined dataframe as a CSV
combined_df.to_csv(output_file, index=False)


# In[4]:


def read_http_log_file(file_path):
    log_to_df = LogToDataFrame()
    try:
        df = log_to_df.create_dataframe(file_path)
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

log_folder = "attacked_data"

http_dataframes = []  # List to store dataframes from HTTP logs

for log_file in os.listdir(log_folder):
    if log_file.endswith(".log") and "http" in log_file.lower():
        full_log_path = os.path.join(log_folder, log_file)
        df = read_http_log_file(full_log_path)
        if df is not None:
            http_dataframes.append(df)


# In[5]:


def read_http_log_file(file_path):
    log_to_df = LogToDataFrame()
    try:
        df = log_to_df.create_dataframe(file_path)
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

log_folder = "attacked_data"
output_folder = "http_dataframes_csv"  # Specify the output folder path

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

http_dataframes = []  # List to store dataframes from HTTP logs

for log_file in os.listdir(log_folder):
    if log_file.endswith(".log") and "http" in log_file.lower():
        full_log_path = os.path.join(log_folder, log_file)
        df = read_http_log_file(full_log_path)
        if df is not None:
            http_dataframes.append(df)
            output_file_path = os.path.join(output_folder, f"{log_file.replace('.log', '')}.csv")
            df.to_csv(output_file_path, index=False)


# In[6]:


try:
    data = pd.read_csv("combined_attack.csv", usecols=['uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p',
                                                       'trans_depth', 'version', 'request_body_len', 'response_body_len',
                                                       'status_code', 'status_msg', 'tags'])
    # Process the data here
    
except KeyError as e:
    print(f"Error reading columns: {e}")
    data = None  # Set data to None to indicate that the read failed

# Now you can check if data is None or not before proceeding further
if data is not None:
    # Process the data further
    pass
else:
    # Handle the case where data could not be read due to missing columns
    pass


# In[7]:


model = tf.keras.models.load_model('new_model6.h5')


# In[8]:


data=data.dropna()


# In[9]:


numerical_features = data.select_dtypes(include=['float64', 'int64']).columns


# Preprocessing - Numerical features (Standard Scaling)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[numerical_features])


# In[10]:


test_data = scaled_features
test_data = scaled_features[:, :7]
predictions = model.predict(test_data)


# In[11]:


mse_loss = np.mean(np.square(test_data - predictions), axis=1)


# In[12]:


threshold = 0.05


# In[13]:


anomalies = mse_loss > threshold


# In[14]:


plt.figure(figsize=(10, 6))
plt.plot(mse_loss, label='Reconstruction Loss')
plt.plot(np.where(anomalies, mse_loss, None), 'ro', label='Anomalies')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.xlabel('Sample Index')
plt.ylabel('Reconstruction Loss (MSE)')
plt.title('Anomaly Detection using Autoencoders')
plt.legend()
plt.show()


# In[ ]:




