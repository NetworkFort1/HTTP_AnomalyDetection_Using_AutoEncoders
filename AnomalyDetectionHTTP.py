import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler, LabelEncoder
from zat.log_to_dataframe import LogToDataFrame

# Define the path to the log file
log_file_path = '/opt/zeek/spool/zeek/http.log'

# Initialize a LogToDataFrame instance
log_to_df = LogToDataFrame()

# Define column names for your DataFrame
columns = ['ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p', 'trans_depth', 'method',
           'host', 'uri', 'referrer', 'version', 'user_agent', 'origin', 'request_body_len',
           'response_body_len', 'status_code', 'status_msg', 'info_code', 'info_msg', 'tags',
           'username', 'password', 'proxied', 'orig_fuids', 'orig_filenames', 'orig_mime_types',
           'resp_fuids', 'resp_filenames', 'resp_mime_types']

# Read the log data into a DataFrame, skip comments, and use the defined column names
df = pd.read_csv(log_file_path, sep="\t", comment="#", header=None, names=columns)

# Drop unnecessary columns
columns_to_drop = ['ts', 'trans_depth', 'host', 'uri', 'uid', 'id.orig_h', 'id.resp_h',
                   'version', 'status_code', 'info_msg', 'tags', 'status_msg', 'username',
                   'password', 'proxied', 'orig_fuids', 'orig_filenames', 'orig_mime_types',
                   'resp_fuids', 'resp_filenames', 'resp_mime_types']
df = df.drop(columns=columns_to_drop)

# Drop rows with missing values
df = df.dropna()

# Load a pre-trained model
model = tf.keras.models.load_model('anomalyDetection.h5')

# Identify numerical and categorical features
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
categorical_features = df.select_dtypes(include=['object']).columns

# Encode categorical features using LabelEncoder
label_encoder = LabelEncoder()
for feature in categorical_features:
    df[feature] = label_encoder.fit_transform(df[feature])

# Preprocessing - Standard Scaling for numerical features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df)

# Make predictions using the loaded model
test_data = scaled_features
predictions = model.predict(test_data)

# Calculate Mean Squared Error (MSE) loss for each data point
mse_loss = np.mean(np.square(test_data - predictions), axis=1)

# Define a threshold for anomaly detection
threshold = 0.5

# Identify anomalies based on MSE loss
anomalies = mse_loss > threshold

df['anomalies']=anomalies

print('Prediction Complete Successfully ')
df.to_csv('Anomalies_HTTP_model_prediction.csv')
print('Prediction Saved Successfully ')



# Create a plot to visualize anomalies
plt.figure(figsize=(10, 6))
plt.plot(mse_loss, label='Reconstruction Loss')
plt.plot(np.where(anomalies, mse_loss, None), 'ro', label='Anomalies')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.xlabel('Sample Index')
plt.ylabel('Reconstruction Loss (MSE)')
plt.title('Anomaly Detection using Autoencoders')
plt.legend()
plt.show()
