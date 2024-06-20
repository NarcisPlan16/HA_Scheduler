import math
import requests
import joblib

import forcaster as forecast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV


ha_url = "http://192.168.0.110:8123"
bearer_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJlYzZhYjAxYTVkM2M0OGE3YjU0OGQ1NmYxNjQyNWQ2ZCIsImlhdCI6MTcxMzM1MDQxNSwiZXhwIjoyMDI4NzEwNDE1fQ.Eutl8pls09_KCIWESOv17gmIzu-RW32eazbHp2V4Wr0"

headers = {
    "Authorization": f"Bearer {bearer_token}",
    "Content-Type": "application/json",
}

lat = "41.963138"
lon = "2.831640"
ini = "2024-01-01"
end = "2024-06-01"  # Year - month - Day


def Start(request_to_api):

    if request_to_api:

        entity = "sensor.smart_meter_63a_energia_real_consumida"
        # o bé sumar els grocs i vermells de la visualització principal. CONSUM_PLACA_A_LO_51, ...
        response = requests.get(
            f"{ha_url}/api/history/period/" + ini + "T00:00:00?end_time=" + end + "T00:00:00&filter_entity_id=" + entity,
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

        data.to_json('LABConsumption.json', orient='split', compression='infer', index='true')

    else:
        data = pd.read_json('LABConsumption.json', orient='split', compression='infer')

    # Add the weather forecast. Lat and lon of UdG p4's building
    lat = "41.963138"
    lon = "2.831640"
    if request_to_api:
        url = f"https://archive-api.open-meteo.com/v1/era5?latitude={lat}&longitude={lon}&start_date={ini}&end_date={end}&hourly=temperature_2m,relativehumidity_2m,dewpoint_2m,apparent_temperature,precipitation,rain,weathercode,pressure_msl,surface_pressure,cloudcover,cloudcover_low,cloudcover_mid,cloudcover_high,et0_fao_evapotranspiration,vapor_pressure_deficit,windspeed_10m,windspeed_100m,winddirection_10m,winddirection_100m,windgusts_10m,shortwave_radiation_instant,direct_radiation_instant,diffuse_radiation_instant,direct_normal_irradiance_instant,terrestrial_radiation_instant"
        response = requests.get(url).json()
        meteo_data = pd.DataFrame(response['hourly'])
        meteo_data = meteo_data.rename(columns={'time': 'Timestamp'})

        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        data['Timestamp'] = data['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        meteo_data['Timestamp'] = pd.to_datetime(meteo_data['Timestamp'])
        meteo_data['Timestamp'] = meteo_data['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

        data = pd.merge(data, meteo_data, on='Timestamp', how='inner')
        data.set_index(data['Timestamp'], inplace=True)
        data = data.drop(columns='Timestamp', axis=1)

        data.to_json('Data_Plus_MeteoForecast.json', orient='split', compression='infer', index=True)
    else:
        data = pd.read_json('Data_Plus_MeteoForecast.json', orient='split', compression='infer')

    data = pd.read_json('Data_Plus_MeteoForecast.json', orient='split', compression='infer')
    #data.index = pd.to_datetime(data.index)

    forecaster = forecast.Forcaster(debug=True)
    forecaster.create_model(
        data=data,
        y='state',
        look_back={-1: [2, 25]},
        colinearity_remove_level=0.9,
        feature_selection='PCA',
        algorithm=['GBoost'],
        params=None,
        escalat='MINMAX',
        max_time=60
    )
    forecaster.save_model("Consumption_model.joblib")

    cons_model = joblib.load("Consumption_model.joblib")
    cons_forecaster = forecast.Forcaster(debug=True)
    cons_forecaster.db = cons_model

    print(cons_forecaster.forcast(data[0:48]))

    """
    plt.figure(figsize=(10, 6))
    x = [i for i in range(0, y_test.size)]
    plt.scatter(x, y_test, color='blue', label='Real', marker='.')
    plt.scatter(x, y_pred, color='orange', label='Predicted', marker='.')
    plt.xlabel('Hours')
    plt.ylabel('Consumption')
    plt.title('Predicted consumption (Kwh)')
    plt.legend()
    plt.show()

    #plt.figure(figsize=(10, 6))
    #x = [i for i in range(0, y_train.size)]
    #plt.scatter(x, y_train, color='blue', label='y_train', marker='.')
    #plt.xlabel('Hours')
    #plt.ylabel('Consumption data (Kwh)')
    #plt.title('Train data')
    #plt.show()

    return y_pred
    """


"""
#---------Test to get the electricity price forecast---------#
tomorrow = datetime.today() + timedelta(1)
tomorrow_str = tomorrow.strftime('%Y%m%d')
url = f"https://www.omie.es/es/file-download?parents%5B0%5D=marginalpdbc&filename=marginalpdbc_{tomorrow_str}.1"

response = requests.get(url)
if response.status_code != "200":
    today = datetime.today().strftime('%Y%m%d')
    url = f"https://www.omie.es/es/file-download?parents%5B0%5D=marginalpdbc&filename=marginalpdbc_{today}.1"
    response = requests.get(url)

with open("omie_price_pred.csv", 'wb') as f:
    f.write(response.content)

hourly_prices = []
with open('omie_price_pred.csv', 'r') as file:
    for line in file.readlines()[1:-1]:
        components = line.strip().split(';')
        components.pop(-1)  # delete blank character at the end
        hourly_price = float(components[-1])
        hourly_prices.append(hourly_price)

print(hourly_prices)
#-----------------------------------------------------------#
"""

"""
url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&forecast_days=2&hourly=temperature_2m,relativehumidity_2m,dewpoint_2m,apparent_temperature,precipitation,rain,weathercode,pressure_msl,surface_pressure,cloudcover,cloudcover_low,cloudcover_mid,cloudcover_high,et0_fao_evapotranspiration,vapor_pressure_deficit,windspeed_10m,windspeed_100m,winddirection_10m,winddirection_100m,windgusts_10m,shortwave_radiation_instant,direct_radiation_instant,diffuse_radiation_instant,direct_normal_irradiance_instant,terrestrial_radiation_instant"
response = requests.get(url).json()
meteo_data = pd.DataFrame(response['hourly'])
meteo_data = meteo_data.rename(columns={'time': 'Timestamp'})

meteo_data['Timestamp'] = pd.to_datetime(meteo_data['Timestamp'])
meteo_data['Timestamp'] = meteo_data['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
meteo_data['Timestamp'] = pd.to_datetime(meteo_data['Timestamp'])

meteo_data['Year'] = meteo_data['Timestamp'].dt.year
meteo_data['Month'] = meteo_data['Timestamp'].dt.month
meteo_data['Day'] = meteo_data['Timestamp'].dt.day
meteo_data['Hour'] = meteo_data['Timestamp'].dt.hour

meteo_data.drop(columns=['Timestamp'], inplace=True)

tomorrow = datetime.today() + timedelta(1)
meteo_data = meteo_data[meteo_data['Day'] == tomorrow.day]
meteo_data.reset_index(drop=True, inplace=True)

meteo_data.to_json('MeteoForecastData.json', orient='split', compression='infer', index=True)
"""

Start(True)




