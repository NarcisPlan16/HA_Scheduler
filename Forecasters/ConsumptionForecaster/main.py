import math
import requests
import joblib
import forcaster as forecast  # Import the forecasting module
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV

# Define Home Assistant URL and the Bearer token for API access
ha_url = "http://192.168.0.117:8123"
bearer_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJlYzZhYjAxYTVkM2M0OGE3YjU0OGQ1NmYxNjQyNWQ2ZCIsImlhdCI6MTcxMzM1MDQxNSwiZXhwIjoyMDI4NzEwNDE1fQ.Eutl8pls09_KCIWESOv17gmIzu-RW32eazbHp2V4Wr0"

# Set headers for the API request
headers = {
    "Authorization": f"Bearer {bearer_token}",  # Include bearer token for authentication
    "Content-Type": "application/json",  # Set content type to JSON
}

# Define geographical coordinates and the date range for data collection
lat = "41.963138"  # Latitude
lon = "2.831640"   # Longitude
ini = "2024-01-01"  # Start date
end = "2024-06-01"  # End date (Year - Month - Day)


def Start(request_to_api):
    """
    Function to retrieve data from Home Assistant and weather forecast APIs.
    
    :param request_to_api: Boolean to decide if data should be fetched from the API.
    :return: Forecasted consumption values.
    """

    if request_to_api:
        # Define the entity to be queried from Home Assistant
        entity = "sensor.smart_meter_63a_energia_real_consumida" # o bé sumar els grocs i vermells de la visualització principal. CONSUM_PLACA_A_LO_51, ...
        
        # Send a GET request to Home Assistant's history API to fetch data
        response = requests.get(
            f"{ha_url}/api/history/period/" + ini + "T00:00:00?end_time=" + end + "T00:00:00&filter_entity_id=" + entity,
            headers=headers)

        # Parse the response JSON and initialize a DataFrame
        response_data = response.json()[0]
        data = pd.DataFrame()
        data = data.from_dict(response_data)
        
        # Define the desired columns to retain
        columns = ['last_updated', 'state']

        # Drop unwanted columns, keeping only the desired ones
        columns_to_drop = [col for col in data.columns if col not in columns]
        data = data.drop(columns=columns_to_drop, axis=1)
        
        # Convert state column to numeric, coercing errors
        data['state'] = data['state'].apply(pd.to_numeric, errors='coerce')
        
        # Process last_updated timestamp
        data['last_updated'] = data['last_updated'].str.split('+').str[0]
        data['last_updated'] = data['last_updated'].apply(datetime.fromisoformat)

        # Set the last_updated column as the index and resample data
        data = data.set_index('last_updated')
        data = data.resample('1s').mean().ffill().resample('1h').mean()
        
        # Store the resampled data's index in a new Timestamp column
        update_indices = data.index
        data['Timestamp'] = update_indices
        
        # Reset index and save data to a JSON file
        data = data.reset_index(drop=True, inplace=False)
        data.to_json('LABConsumption.json', orient='split', compression='infer', index='true')

    else:
        # Load data from JSON file if not fetching from API
        data = pd.read_json('LABConsumption.json', orient='split', compression='infer')

    # Add the weather forecast data based on latitude and longitude
    if request_to_api:
        url = f"https://archive-api.open-meteo.com/v1/era5?latitude={lat}&longitude={lon}&start_date={ini}&end_date={end}&hourly=temperature_2m,relativehumidity_2m,dewpoint_2m,apparent_temperature,precipitation,rain,weathercode,pressure_msl,surface_pressure,cloudcover,cloudcover_low,cloudcover_mid,cloudcover_high,et0_fao_evapotranspiration,vapor_pressure_deficit,windspeed_10m,windspeed_100m,winddirection_10m,winddirection_100m,windgusts_10m,shortwave_radiation_instant,direct_radiation_instant,diffuse_radiation_instant,direct_normal_irradiance_instant,terrestrial_radiation_instant"
        
        # Fetch weather data from Open Meteo API
        response = requests.get(url).json()
        meteo_data = pd.DataFrame(response['hourly'])
        meteo_data = meteo_data.rename(columns={'time': 'Timestamp'})  # Rename timestamp column for merging

        # Convert Timestamp columns to datetime format for merging
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        data['Timestamp'] = data['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        meteo_data['Timestamp'] = pd.to_datetime(meteo_data['Timestamp'])
        meteo_data['Timestamp'] = meteo_data['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # Merge consumption data with weather data
        data = pd.merge(data, meteo_data, on='Timestamp', how='inner')
        data.set_index(data['Timestamp'], inplace=True)  # Set Timestamp as the index
        data = data.drop(columns='Timestamp', axis=1)  # Drop the Timestamp column

        # Save the merged data to a new JSON file
        data.to_json('Data_Plus_MeteoForecast.json', orient='split', compression='infer', index=True)
    else:
        # Load data with weather forecast if not fetching from API
        data = pd.read_json('Data_Plus_MeteoForecast.json', orient='split', compression='infer')

    #data = pd.read_json('Data_Plus_MeteoForecast.json', orient='split', compression='infer')
    #data.index = pd.to_datetime(data.index)

    # Create and save the forecasting model
    forecaster = forecast.Forcaster(debug=True)
    forecaster.create_model(
        data=data,
        y='state',  # Target variable for forecasting
        look_back={-1: [2, 25]},  # Look-back periods for historical data
        colinearity_remove_level=0.9,  # Level for removing collinear features
        feature_selection='PCA',  # Principal Component Analysis for feature selection
        algorithm=['GBoost'],  # Use Gradient Boosting algorithm
        params=None,  # Parameters can be defined for model tuning
        escalat='MINMAX',  # Use min-max scaling for feature scaling
        max_time=60  # Set maximum time for model creation
    )
    forecaster.save_model("Consumption_model.joblib")  # Save the model to a file

    # Load the saved model for forecasting
    cons_model = joblib.load("Consumption_model.joblib")
    cons_forecaster = forecast.Forcaster(debug=True)
    cons_forecaster.db = cons_model  # Assign loaded model to forecaster

    # Print the forecasted values for the next 48 hours
    print(cons_forecaster.forcast(data[0:48]))

    # The following commented code is for visualization (if needed)
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
tomorrow = datetime.today() + timedelta(1)  # Get tomorrow's date
tomorrow_str = tomorrow.strftime('%Y%m%d')  # Format date for URL
url = f"https://www.omie.es/es/file-download?parents%5B0%5D=marginalpdbc&filename=marginalpdbc_{tomorrow_str}.1"  # URL to fetch tomorrow's price

response = requests.get(url)  # Send GET request
if response.status_code != "200":  # Check if response is valid
    today = datetime.today().strftime('%Y%m%d')  # Format today's date
    url = f"https://www.omie.es/es/file-download?parents%5B0%5D=marginalpdbc&filename=marginalpdbc_{today}.1"  # URL to fetch today's price
    response = requests.get(url
