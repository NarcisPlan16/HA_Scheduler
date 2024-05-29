import pandas as pd
import requests
import joblib
import os

from datetime import datetime, timedelta

current_dir = os.getcwd()
prod_model = joblib.load(current_dir + "/Abstraction/Forecaster Models/Generation_model.joblib")
cons_model = joblib.load(current_dir + "/Abstraction/Forecaster Models/Consumption_model.joblib")


def obtainMeteoData(latitude, longitude):

    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&forecast_days=1&hourly=temperature_2m,relativehumidity_2m,dewpoint_2m,apparent_temperature,precipitation,rain,weathercode,pressure_msl,surface_pressure,cloudcover,cloudcover_low,cloudcover_mid,cloudcover_high,et0_fao_evapotranspiration,vapor_pressure_deficit,windspeed_10m,windspeed_100m,winddirection_10m,winddirection_100m,windgusts_10m,shortwave_radiation_instant,direct_radiation_instant,diffuse_radiation_instant,direct_normal_irradiance_instant,terrestrial_radiation_instant"
    response = requests.get(url).json()
    meteo_data = pd.DataFrame(response['hourly'])
    meteo_data = meteo_data.rename(columns={'time': 'Timestamp'})

    meteo_data['Timestamp'] = pd.to_datetime(meteo_data['Timestamp'])
    meteo_data['Timestamp'] = meteo_data['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    meteo_data.to_json('MeteoForecastData.json', orient='split', compression='infer', index=True)

    return meteo_data


def predictConsumption(meteo_data, scheduling_data):
    # Predict the consumption taking into account the scheduling of the assets
    # The scheduling data must contain only the columns ['Year', 'Month', 'Day', 'Hour']

    data = pd.merge(meteo_data, scheduling_data, on=['Year', 'Month', 'Day', 'Hour'], how='inner')
    consumption = cons_model.predict(data)

    return consumption


def predictProduction(meteo_data, scheduling_data):
    # Predict the production taking into account the assets and its schedule
    # The scheduling data must contain only the columns ['Year', 'Month', 'Day', 'Hour']

    data = pd.merge(meteo_data, scheduling_data, on=['Year', 'Month', 'Day', 'Hour'], how='inner')
    production = prod_model.predict(data)

    return production

