import pandas as pd
import requests
import joblib
import os

import forecaster as forecast
from datetime import datetime, timedelta

# Create the prediction and the consumption forecasters from the given models
current_dir = os.getcwd()
prod_model = joblib.load(current_dir + "/Abstraction/Forecaster Models/Generation_model.joblib")
prod_forecaster = forecast.forecaster(debug=True)
prod_forecaster.db = prod_model

cons_model = joblib.load(current_dir + "/Abstraction/Forecaster Models/Consumption_model.joblib")
cons_forecaster = forecast.forecaster(debug=True)
cons_forecaster.db = cons_model

def obtainMeteoData(latitude, longitude):
    """
    Obtains the meteo data forecast for the next day and today's data of the specified latitude and longitude.

    Parameters
    -----------
    latitude : float
    longitude : float
    -----------
    Returns
    -----------
    Returns a DataFrame with the meteo data of size (48, n).
    """

    today = datetime.today().strftime('%Y-%m-%d')
    tomorrow = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&start_date={today}&end_date={tomorrow}&hourly=temperature_2m,relativehumidity_2m,dewpoint_2m,apparent_temperature,precipitation,rain,weathercode,pressure_msl,surface_pressure,cloudcover,cloudcover_low,cloudcover_mid,cloudcover_high,et0_fao_evapotranspiration,vapor_pressure_deficit,windspeed_10m,windspeed_100m,winddirection_10m,winddirection_100m,windgusts_10m,shortwave_radiation_instant,direct_radiation_instant,diffuse_radiation_instant,direct_normal_irradiance_instant,terrestrial_radiation_instant"
    response = requests.get(url).json()
    meteo_data = pd.DataFrame(response['hourly'])
    meteo_data = meteo_data.rename(columns={'time': 'Timestamp'})

    meteo_data['Timestamp'] = pd.to_datetime(meteo_data['Timestamp'])
    meteo_data['Timestamp'] = meteo_data['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    meteo_data['Timestamp'] = pd.to_datetime(meteo_data['Timestamp'])

    return meteo_data


def predictConsumption(meteo_data: pd.DataFrame, scheduling_data: pd.DataFrame):
    """
    Predict the consumption taking into account the active hours scheduled of the assets

    Parameters
    -----------
    meteo_data : DataFrame
        Pandas dataframe with the meteorological data prediction for the next day and the data of today. 
        For example, if predicting the next 24 hours, this parameter must have size (48, n).
    scheduling_data : DataFrame
        Pandas dataframe with the scheduling data relative to the assets that will be optimized for consumption. This dataframe must contain only
        relevant attributes. For now only supported attribute "state". On future work, more attributes could be considered as the model is being updated.
        This parameter must have size (48, m).
    -----------
    Returns
    -----------
    Returns a DataFrame with the consumption prediction with size (24, n + m).
    """ 

    meteo_data['Timestamp'] = pd.to_datetime(meteo_data['Timestamp'])
    scheduling_data['Timestamp'] = pd.to_datetime(scheduling_data['Timestamp'])
    data = pd.merge(scheduling_data, meteo_data, on=['Timestamp'], how='inner')
    data = data.set_index('Timestamp')
    data.index = pd.to_datetime(data.index)

    consumption = cons_forecaster.forcast(data)
    print("--------------------CONSUMPTION PREDICTION DONE--------------------")

    return consumption


def predictProduction(meteo_data: pd.DataFrame, scheduling_data: pd.DataFrame):
    """
    Predict the production taking into account the active hours scheduled of the assets

    Parameters
    -----------
    meteo_data : DataFrame
        Pandas dataframe with the meteorological data prediction for the next day and the data of today. 
        For example, if predicting the next 24 hours, this parameter must have size (48, n).
    scheduling_data : DataFrame
        Pandas dataframe with the scheduling data relative to the assets that will be optimized for production. This dataframe must contain only
        relevant attributes. For now only supported attribute "state". On future work, more attributes could be considered as the model is being updated.
        This parameter must have size (48, m).
    -----------
    Returns
    -----------
    Returns a DataFrame with the production prediction with size (24, n + m).
    """

    meteo_data['Timestamp'] = pd.to_datetime(meteo_data['Timestamp'])
    scheduling_data['Timestamp'] = pd.to_datetime(scheduling_data['Timestamp'])
    data = pd.merge(scheduling_data, meteo_data, on=['Timestamp'], how='inner')
    data = data.set_index('Timestamp')
    data.index = pd.to_datetime(data.index)

    production = prod_forecaster.forcast(data)
    print("--------------------PRODUCTION PREDICTION DONE--------------------")

    return production

