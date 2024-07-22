import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import pickle

# Load data
url = 'https://drive.google.com/file/d/1ZznG8sKL46fVAneRY7PlzPs50z75L-Ok/view?usp=drive_link'
data = pd.read_csv(url)

# Drop unnecessary columns
data = data.drop(['Open', 'Low', 'Close', 'Adj Close', 'Volume'], axis='columns')

# Convert 'Date' to datetime and set it as index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Initialize the 'High' column as time series data
ts = data['High']

# Train the ARIMA model
model_ARIMA = ARIMA(ts, order=(1, 1, 1))
result_ARIMA = model_ARIMA.fit()

# Save the trained model to a file
with open('model/arima_model.pkl', 'wb') as file:
    pickle.dump(result_ARIMA, file)

print("Model trained and saved successfully.")
