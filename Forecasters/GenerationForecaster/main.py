import math
import numpy as np
import requests
import joblib
import forecaster as forecast
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

# Define the Home Assistant API URL and authorization token
ha_url = "http://192.168.0.110:8123"
bearer_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJlYzZhYjAxYTVkM2M0OGE3YjU0OGQ1NmYxNjQyNWQ2ZCIsImlhdCI6MTcxMzM1MDQxNSwiZXhwIjoyMDI4NzEwNDE1fQ.Eutl8pls09_KCIWESOv17gmIzu-RW32eazbHp2V4Wr0"

# Set the headers for API requests
headers = {
    "Authorization": f"Bearer {bearer_token}",
    "Content-Type": "application/json",
}

# Define the date range for data collection
ini = "2022-01-01"
end = "2024-04-16"  # Year - month - Day
request_to_api = True  # Flag to indicate whether to request data from the API

# Request data from the Home Assistant API
if request_to_api:
    entity = "sensor.symo_3_7_3_s_1_alimentacion_ca"  # Sensor entity to fetch data from
    response = requests.get(f"{ha_url}/api/history/period/"+ini+"T00:00:00?end_time="+end+"T23:00:00&filter_entity_id="+entity, headers=headers)

    # Parse the JSON response
    response_data = response.json()[0]
    data = pd.DataFrame()  # Initialize an empty DataFrame
    data = data.from_dict(response_data)  # Convert the response data to a DataFrame
    columns = ['last_updated', 'state']  # Define the columns to keep

    # Drop all columns except the desired ones
    columns_to_drop = [col for col in data.columns if col not in columns]
    data = data.drop(columns=columns_to_drop, axis=1)
    
    # Convert 'state' to numeric and process 'last_updated' column
    data['state'] = data['state'].apply(pd.to_numeric, errors='coerce')
    data['last_updated'] = data['last_updated'].str.split('+').str[0]
    data['last_updated'] = data['last_updated'].apply(datetime.fromisoformat)

    # Set 'last_updated' as the DataFrame index
    data = data.set_index('last_updated')
    
    # Resample data to hourly averages
    data = data.resample('1s').mean().ffill().resample('1h').mean()
    update_indices = data.index  # Get updated indices
    data['Timestamp'] = update_indices  # Add a Timestamp column
    data = data.reset_index(drop=True, inplace=False)  # Reset index

    # Save the processed data to a JSON file
    data.to_json('PVProduction.json', orient='split', compression='infer', index='true')

else:
    # If not requesting from the API, load data from the saved JSON file
    data = pd.read_json('PVProduction.json', orient='split', compression='infer')

# Latitude and longitude for weather data fetching. Lat and lon of UdG p4's building
lat = "41.963138"
lon = "2.831640"

# Fetch weather forecast data if request_to_api is True
if request_to_api:
    # Build the URL for weather data API request
    url = f"https://archive-api.open-meteo.com/v1/era5?latitude={lat}&longitude={lon}&start_date={ini}&end_date={end}&hourly=temperature_2m,relativehumidity_2m,dewpoint_2m,apparent_temperature,precipitation,rain,weathercode,pressure_msl,surface_pressure,cloudcover,cloudcover_low,cloudcover_mid,cloudcover_high,et0_fao_evapotranspiration,vapor_pressure_deficit,windspeed_10m,windspeed_100m,winddirection_10m,winddirection_100m,windgusts_10m,shortwave_radiation_instant,direct_radiation_instant,diffuse_radiation_instant,direct_normal_irradiance_instant,terrestrial_radiation_instant"
    response = requests.get(url).json()  # Make the API request and parse the response
    meteo_data = pd.DataFrame(response['hourly'])  # Create a DataFrame from the hourly weather data
    meteo_data = meteo_data.rename(columns={'time': 'Timestamp'})  # Rename the 'time' column

    # Process 'Timestamp' in both dataframes
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data['Timestamp'] = data['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    meteo_data['Timestamp'] = pd.to_datetime(meteo_data['Timestamp'])
    meteo_data['Timestamp'] = meteo_data['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Merge PV production data with weather data on 'Timestamp'
    data = pd.merge(data, meteo_data, on='Timestamp', how='inner')
    data = data.drop(columns='Timestamp', axis=1)  # Drop the 'Timestamp' column after merging

    # Save the combined data to a JSON file
    data.to_json('Data_Plus_MeteoForecast.json', orient='split', compression='infer', index=True)

else:
    # If not requesting from the API, load the combined data from the saved JSON file
    data = pd.read_json('Data_Plus_MeteoForecast.json', orient='split', compression='infer')

# Load the combined data and set the index to datetime
data = pd.read_json('Data_Plus_MeteoForecast.json', orient='split', compression='infer')
data.index = pd.to_datetime(data.index)

# Initialize the forecaster with debug mode
forecaster = forecast.forecaster(debug=True)

# Create a forecasting model using the combined data
forecaster.create_model(
    data=data,
    y='state',  # Target variable for prediction
    look_back={-1: [2, 25]},  # Define look-back periods for time series data
    colinearity_remove_level=0.9,  # Threshold for collinearity removal
    feature_selection='PCA',  # Feature selection method
    algorithm=['GBoost'],  # Algorithm for model creation
    params=None,  # Additional parameters (if any)
    escalat='MINMAX',  # Scaling method
    max_time=60  # Maximum time allowed for model training
)

# Save the trained model to a file
forecaster.save_model("Generation_model.joblib")

"""
# Uncomment to evaluate the model performance using metrics
y_pred = model.predict(X_test)  # Make predictions on the test set
mse = mean_squared_error(y_test, y_pred)  # Calculate Mean Squared Error
print("MSE: ", mse)  # Print MSE
mape = mean_absolute_percentage_error(y_test, y_pred)  # Calculate Mean Absolute Percentage Error
print("MAPE: ", mape)  # Print MAPE
r2 = r2_score(y_test, y_pred)  # Calculate R² score
print("R2 score: ", r2)  # Print R² score
"""




