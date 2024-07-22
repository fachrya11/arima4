import pandas as pd
import streamlit as st
import requests
import numpy as np
import pickle
from datetime import datetime, timedelta

# Load the trained ARIMA model
with open('model/arima_model.pkl', 'rb') as file:
    model_ARIMA = pickle.load(file)

st.title('Aplikasi Prediksi')

start_date = st.date_input('Tanggal Mulai', value=None)
end_date = st.date_input('Tanggal Akhir', value=None)

def predict(start_date, end_date):
    try:
        # Generate date range for prediction
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Use the ARIMA model to predict the stock prices
        predictions_diff = model_ARIMA.predict(start=len(date_range), end=len(date_range) + (end_date - start_date).days - 1)
        predictions_diff_cumsum = predictions_diff.cumsum()
        last_value = model_ARIMA.fittedvalues[-1]
        predictions = last_value + predictions_diff_cumsum

        # Prepare the results
        results = {
            'date': date_range.strftime('%Y-%m-%d').tolist(),
            'predictions': predictions.tolist()
        }
        
        # Ensure that both lists are of the same length
        if len(results['date']) != len(results['predictions']):
            raise ValueError("Length of dates and predictions must be the same.")
        
        return results
    except Exception as e:
        return {'error': str(e)}

if st.button('Prediksi'):
    if start_date and end_date:
        results = predict(start_date, end_date)
        if 'error' in results:
            st.write('Terjadi kesalahan:', results['error'])
        else:
            # Create DataFrame from results
            df = pd.DataFrame({
                'date': results['date'],
                'predictions': results['predictions']
            })
            st.write('Hasil prediksi:')
            st.line_chart(df.set_index('date'))
    else:
        st.write('Silakan masukkan tanggal mulai dan tanggal akhir.')
