import pandas as pd
import streamlit as st
import numpy as np
import pickle
import requests
from datetime import datetime, timedelta

# Load the trained ARIMA model
with open('model/arima_model.pkl', 'rb') as file:
    model_ARIMA = pickle.load(file)


# Input untuk tanggal mulai dan tanggal akhir
start_date = st.date_input('Tanggal Mulai', value=None)
end_date = st.date_input('Tanggal Akhir', value=None)

# Tombol untuk melakukan prediksi
if st.button('Prediksi'):
    if start_date and end_date:
        # Mengirim permintaan POST ke API Flask dengan data tanggal
        response = requests.post(
            'http://localhost:5000/predict',
            json={"start_date": start_date.isoformat(), "end_date": end_date.isoformat()}
        )
        
        if response.status_code == 200:
            result = response.json()
            st.write('Hasil prediksi:', result.get('prediction', 'Tidak ada hasil'))
        else:
            st.write('Terjadi kesalahan:', response.status_code)
    else:
        st.write('Silakan masukkan tanggal mulai dan tanggal akhir.')

def index():
    return render_template('index.html')

def predict():
    try:
        data = request.get_json(force=True)
        start_date = datetime.strptime(data['start_date'], '%Y-%m-%d')
        end_date = datetime.strptime(data['end_date'], '%Y-%m-%d')
        
        # Generate date range for prediction
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Use the ARIMA model to predict the stock prices
        predictions_diff = model_ARIMA.predict(start=len(date_range), end=len(date_range) + (end_date - start_date).days - 1)
        predictions_diff_cumsum = predictions_diff.cumsum()
        last_value = model_ARIMA.fittedvalues[-1]
        predictions = last_value + predictions_diff_cumsum

        # Prepare the results
        results = {'date': date_range.strftime('%Y-%m-%d').tolist(), 'predictions': predictions.tolist()}
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)})
