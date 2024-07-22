import pandas as pd
import streamlit as st
import numpy as np
import pickle
from datetime import datetime, timedelta

# Load the trained ARIMA model
with open('model/arima_model.pkl', 'rb') as file:
    model_ARIMA = pickle.load(file)

# Form input dari pengguna
input_data = st.text_input('Masukkan data untuk prediksi:')

if st.button('Prediksi'):
    # Mengirim permintaan POST ke API Flask
    response = requests.post('http://127.0.0.1:5000', json={"data": input_data})
    
    if response.status_code == 200:
        result = response.json()
        st.write('Hasil prediksi:', result['prediction'])
    else:
        st.write('Terjadi kesalahan:', response.status_code)

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
