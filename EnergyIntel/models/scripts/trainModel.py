import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import gdown
import zipfile
import os
import pickle

# Google Drive file ID
file_id = '1p3K3KaDR0iCMFJMK0-y_9-WbtcxVqioP'

# Define output file path
output_path = 'data/household_power_consumption.txt'

# Download dataset from Google Drive
gdown.download(f'https://drive.google.com/uc?id={file_id}', output_path, quiet=False)

# Load the dataset
dt = pd.read_csv(output_path, sep=';', parse_dates={'dt': ['Date', 'Time']}, infer_datetime_format=True, low_memory=False, na_values=['nan', '?'], index_col='dt')

# Data preprocessing
dt.replace('?', np.nan, inplace=True)
dt.dropna(how='all', inplace=True)
for col in dt.columns:
    dt[col] = dt[col].astype("float64")

values = dt.values
dt['Sub_metering_4'] = (values[:, 0] * 1000 / 60) - (values[:, 4] + values[:, 5] + values[:, 6])

# Select features and target
X = dt.iloc[:, [1, 3, 4, 5, 6]]
y = dt.iloc[:, 0]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Linear Regression model
lm = LinearRegression()
lm.fit(X_train, y_train)

# Evaluate the model
predictions = lm.predict(X_test)
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print('RSquarevalue:', metrics.r2_score(y_test, predictions))

# Save the trained model to a .pkl file
model_filename = 'models/PCASSS_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(lm, file)
