import pandas as pd
import requests
import joblib
import os

import forcaster as forecast
from datetime import datetime, timedelta

current_dir = os.getcwd()
prod_model = joblib.load(current_dir + "/Abstraction/Forecaster Models/Generation_model.joblib")
prod_forecaster = forecast.Forcaster(debug=True)
prod_forecaster.db = prod_model

cons_model = joblib.load(current_dir + "/Abstraction/Forecaster Models/Consumption_model.joblib")
cons_forecaster = forecast.Forcaster(debug=True)
cons_forecaster.db = cons_model

def obtainMeteoData(latitude, longitude):
    # forecaste_days is 2 because if we set it to 1, the open-meteo api gives us the forcast for today. Instead we ant the forecast for tomorrow.

    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&forecast_days=2&hourly=temperature_2m,relativehumidity_2m,dewpoint_2m,apparent_temperature,precipitation,rain,weathercode,pressure_msl,surface_pressure,cloudcover,cloudcover_low,cloudcover_mid,cloudcover_high,et0_fao_evapotranspiration,vapor_pressure_deficit,windspeed_10m,windspeed_100m,winddirection_10m,winddirection_100m,windgusts_10m,shortwave_radiation_instant,direct_radiation_instant,diffuse_radiation_instant,direct_normal_irradiance_instant,terrestrial_radiation_instant"
    response = requests.get(url).json()
    meteo_data = pd.DataFrame(response['hourly'])
    meteo_data = meteo_data.rename(columns={'time': 'Timestamp'})

    meteo_data['Timestamp'] = pd.to_datetime(meteo_data['Timestamp'])
    meteo_data['Timestamp'] = meteo_data['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    meteo_data['Timestamp'] = pd.to_datetime(meteo_data['Timestamp'])

    tomorrow = datetime.today() + timedelta(days=1)
    start_tomorrow = datetime(tomorrow.year, tomorrow.month, tomorrow.day, 0, 0, 0)
    meteo_data = meteo_data[meteo_data['Timestamp'] >= start_tomorrow]

    today = (datetime.today() - pd.Timedelta(hours=1)).strftime('%Y-%m-%d')
    tomorrow = (datetime.today() + pd.Timedelta(hours=0)).strftime('%Y-%m-%d')
    url = f"https://archive-api.open-meteo.com/v1/era5?latitude={latitude}&longitude={longitude}&start_date={today}&end_date={tomorrow}&hourly=temperature_2m,relativehumidity_2m,dewpoint_2m,apparent_temperature,precipitation,rain,weathercode,pressure_msl,surface_pressure,cloudcover,cloudcover_low,cloudcover_mid,cloudcover_high,et0_fao_evapotranspiration,vapor_pressure_deficit,windspeed_10m,windspeed_100m,winddirection_10m,winddirection_100m,windgusts_10m,shortwave_radiation_instant,direct_radiation_instant,diffuse_radiation_instant,direct_normal_irradiance_instant,terrestrial_radiation_instant"
    response = requests.get(url).json()
    today_meteo_data = pd.DataFrame(response['hourly'])
    today_meteo_data = today_meteo_data.rename(columns={'time': 'Timestamp'})

    today_meteo_data['Timestamp'] = pd.to_datetime(today_meteo_data['Timestamp'])
    today_meteo_data['Timestamp'] = today_meteo_data['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    data = pd.concat([today_meteo_data, meteo_data], ignore_index=True)
    data.reset_index(inplace=True, drop=True)

    return data


def predictConsumption(meteo_data, scheduling_data):
    # Predict the consumption taking into account the scheduling of the assets
    # The scheduling data must contain only the columns ['Year', 'Month', 'Day', 'Hour']

    data = pd.merge(scheduling_data, meteo_data, on=['Timestamp'], how='inner')
    data = data.set_index('Timestamp')
    data.index = pd.to_datetime(data.index)

    consumption = cons_forecaster.forcast(data)
    print("----------------------------------CONSUMPTION PREDICTION DONE----------------------------------")

    return consumption


def predictProduction(meteo_data, scheduling_data):
    # Predict the production taking into account the assets and its schedule
    # The scheduling data must contain only the columns ['Year', 'Month', 'Day', 'Hour']

    data = pd.merge(scheduling_data, meteo_data, on=['Timestamp'], how='inner')
    data = data.set_index('Timestamp')
    data.index = pd.to_datetime(data.index)

    production = prod_forecaster.forcast(data)
    print("----------------------------------PRODUCTION PREDICTION DONE----------------------------------")

    return production

