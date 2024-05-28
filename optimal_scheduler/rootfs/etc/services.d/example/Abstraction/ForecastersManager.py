import pd
import requests
import joblib
import os

from datetime import datetime, timedelta

current_dir = os.getcwd()
prod_model = joblib.load(current_dir + "/Abstraction/Forecaster Models/Generation_model.joblib")
cons_model = joblib.load(current_dir + "/Abstraction/Forecaster Models/Consumption_model.joblib")
today = datetime.today().strftime('%Y%m%d')

def predictConsumptionData(ha_url, headers, latitude, longitude):
    
    entity = "sensor.sonnenbatterie_79259_consumption_w"
    # o bé sumar els grocs i vermells de la visualització principal. CONSUM_PLACA_A_LO_51, ...
    response = requests.get(
        f"{ha_url}/api/history/period/" + today + "T00:00:00?end_time=" + today + "T23:00:00&filter_entity_id=" + entity,
        headers=headers)

    response_data = response.json()[0]
    data = pd.DataFrame()
    data = data.from_dict(response_data)
    columns = ['last_updated', 'state']

    # Drop columns except the desired ones
    columns_to_drop = [col for col in data.columns if col not in columns]
    data = data.drop(columns=columns_to_drop, axis=1)
    data['state'] = data['state'].apply(pd.to_numeric, errors='coerce')
    data['last_updated'] = data['last_updated'].str.split('+').str[0]
    data['last_updated'] = data['last_updated'].apply(datetime.fromisoformat)

    data = data.set_index('last_updated')
    data = data.resample('1s').mean().ffill().resample('1h').mean()
    update_indices = data.index
    data['Timestamp'] = update_indices
    data = data.reset_index(drop=True, inplace=False)
    data['Year'] = data['Timestamp'].dt.year
    data['Month'] = data['Timestamp'].dt.month
    data['Day'] = data['Timestamp'].dt.day
    data['Hour'] = data['Timestamp'].dt.hour

    url = f"https://archive-api.open-meteo.com/v1/era5?latitude={latitude}&longitude={longitude}&start_date={today}&end_date={today}&hourly=temperature_2m,relativehumidity_2m,dewpoint_2m,apparent_temperature,precipitation,rain,weathercode,pressure_msl,surface_pressure,cloudcover,cloudcover_low,cloudcover_mid,cloudcover_high,et0_fao_evapotranspiration,vapor_pressure_deficit,windspeed_10m,windspeed_100m,winddirection_10m,winddirection_100m,windgusts_10m,shortwave_radiation_instant,direct_radiation_instant,diffuse_radiation_instant,direct_normal_irradiance_instant,terrestrial_radiation_instant"
    response = requests.get(url).json()
    meteo_data = pd.DataFrame(response['hourly'])
    meteo_data = meteo_data.rename(columns={'time': 'Timestamp'})

    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data['Timestamp'] = data['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    meteo_data['Timestamp'] = pd.to_datetime(meteo_data['Timestamp'])
    meteo_data['Timestamp'] = meteo_data['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    data = pd.merge(data, meteo_data, on='Timestamp', how='inner')
    data = data.drop(columns='Timestamp', axis=1)

    data.to_json('Data_Plus_MeteoForecast.json', orient='split', compression='infer', index=True)


def predictProductionData( ha_url, headers, latitude, longitude):
    
    entity = "sensor.sonnenbatterie_79259_meter_production_4_1_w_total"
    response = requests.get(f"{ha_url}/api/history/period/"+today+"T00:00:00?end_time="+today+"T23:00:00&filter_entity_id="+entity, headers=headers)

    response_data = response.json()[0]
    data = pd.DataFrame()
    data = data.from_dict(response_data)
    columns = ['last_updated', 'state']

    # Drop columns except the desired ones
    columns_to_drop = [col for col in data.columns if col not in columns]
    data = data.drop(columns=columns_to_drop, axis=1)
    data['state'] = data['state'].apply(pd.to_numeric, errors='coerce')
    data['last_updated'] = data['last_updated'].str.split('+').str[0]
    data['last_updated'] = data['last_updated'].apply(datetime.fromisoformat)

    data = data.set_index('last_updated')
    data = data.resample('1s').mean().ffill().resample('1h').mean()
    update_indices = data.index
    data['Timestamp'] = update_indices
    data = data.reset_index(drop=True, inplace=False)
    data['Year'] = data['Timestamp'].dt.year
    data['Month'] = data['Timestamp'].dt.month
    data['Day'] = data['Timestamp'].dt.day
    data['Hour'] = data['Timestamp'].dt.hour

    url = f"https://archive-api.open-meteo.com/v1/era5?latitude={latitude}&longitude={longitude}&start_date={today}&end_date={today}&hourly=temperature_2m,relativehumidity_2m,dewpoint_2m,apparent_temperature,precipitation,rain,weathercode,pressure_msl,surface_pressure,cloudcover,cloudcover_low,cloudcover_mid,cloudcover_high,et0_fao_evapotranspiration,vapor_pressure_deficit,windspeed_10m,windspeed_100m,winddirection_10m,winddirection_100m,windgusts_10m,shortwave_radiation_instant,direct_radiation_instant,diffuse_radiation_instant,direct_normal_irradiance_instant,terrestrial_radiation_instant"
    response = requests.get(url).json()
    meteo_data = pd.DataFrame(response['hourly'])
    meteo_data = meteo_data.rename(columns={'time': 'Timestamp'})

    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data['Timestamp'] = data['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    meteo_data['Timestamp'] = pd.to_datetime(meteo_data['Timestamp'])
    meteo_data['Timestamp'] = meteo_data['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    data = pd.merge(data, meteo_data, on='Timestamp', how='inner')
    data = data.drop(columns='Timestamp', axis=1)

    data.to_json('Data_Plus_MeteoForecast.json', orient='split', compression='infer', index=True)

