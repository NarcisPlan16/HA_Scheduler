import math

import requests
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

ha_url = "http://192.168.0.110:8123"
bearer_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJlYzZhYjAxYTVkM2M0OGE3YjU0OGQ1NmYxNjQyNWQ2ZCIsImlhdCI6MTcxMzM1MDQxNSwiZXhwIjoyMDI4NzEwNDE1fQ.Eutl8pls09_KCIWESOv17gmIzu-RW32eazbHp2V4Wr0"

headers = {
    "Authorization": f"Bearer {bearer_token}",
    "Content-Type": "application/json",
}

ini = "2023-01-01"
end = "2024-04-16"  # Year - month - Day
request_to_api = True
if request_to_api:

    entity = "sensor.sonnenbatterie_79259_consumption_w"
    # o bé sumar els grocs i vermells de la visualització principal. CONSUM_PLACA_A_LO_51, ...
    response = requests.get(f"{ha_url}/api/history/period/"+ini+"T00:00:00?end_time="+end+"T00:00:00&filter_entity_id="+entity, headers=headers)

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

    data.to_json('LABConsumption.json', orient='split', compression='infer', index='true')

else:
    data = pd.read_json('LABConsumption.json', orient='split', compression='infer')

# Add the weather forecast. Lat and lon of UdG p4's building
lat = "41.963138"
lon = "2.831640"
url = f"https://archive-api.open-meteo.com/v1/era5?latitude={lat}&longitude={lon}&start_date={ini}&end_date={end}&hourly=temperature_2m,relativehumidity_2m,dewpoint_2m,apparent_temperature,precipitation,rain,weathercode,pressure_msl,surface_pressure,cloudcover,cloudcover_low,cloudcover_mid,cloudcover_high,et0_fao_evapotranspiration,vapor_pressure_deficit,windspeed_10m,windspeed_100m,winddirection_10m,winddirection_100m,windgusts_10m,shortwave_radiation_instant,direct_radiation_instant,diffuse_radiation_instant,direct_normal_irradiance_instant,terrestrial_radiation_instant"
response = requests.get(url).json()
meteo_data = pd.DataFrame(response['hourly'])
meteo_data = meteo_data.rename(columns={'time': 'Timestamp'})

data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data['Timestamp'] = data['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
meteo_data['Timestamp'] = pd.to_datetime(meteo_data['Timestamp'])
meteo_data['Timestamp'] = meteo_data['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

data = pd.merge(data, meteo_data, on='Timestamp', how='inner')
data = data.drop(columns='Timestamp', axis=1)

print("Preprocessing done")
print(data)