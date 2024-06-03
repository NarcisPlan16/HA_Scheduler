import pandas as pd
import requests
import joblib
import os

from datetime import datetime, timedelta

current_dir = os.getcwd()
prod_model = joblib.load(current_dir + "/Abstraction/Forecaster Models/Generation_model.joblib")
cons_model = joblib.load(current_dir + "/Abstraction/Forecaster Models/Consumption_model.joblib")


def obtainMeteoData(latitude, longitude):
    # forecaste_days is 2 because if we set it to 1, the open-meteo api gives us the forcast for today. Instead we ant the forecast for tomorrow.

    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&forecast_days=2&hourly=temperature_2m,relativehumidity_2m,dewpoint_2m,apparent_temperature,precipitation,rain,weathercode,pressure_msl,surface_pressure,cloudcover,cloudcover_low,cloudcover_mid,cloudcover_high,et0_fao_evapotranspiration,vapor_pressure_deficit,windspeed_10m,windspeed_100m,winddirection_10m,winddirection_100m,windgusts_10m,shortwave_radiation_instant,direct_radiation_instant,diffuse_radiation_instant,direct_normal_irradiance_instant,terrestrial_radiation_instant"
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

    return meteo_data

def GroupInstances(input_data: pd.DataFrame, start_date, end_date):

    res = []

    date = start_date
    index = 0
    while start_date <= date < end_date and index < input_data.__len__():

        row = input_data.iloc[index]
        res.append(row)
        date = pd.Timestamp(year=int(row["Year"]), month=int(row["Month"]), day=int(row["Day"]), hour=int(row["Hour"]))
        index += 1

    return res, len(res)


def GenerateNewColumns(data_dict: dict):

    key, row = next(iter(data_dict.items()))
    new_columns = []

    index = 0
    for element in row:
        for key_name, value in element.items():
            new_columns.append(key_name + "_" + str(index))
        index += 1

    return new_columns

def PrepareBatches(input_data: pd.DataFrame, timeframe: str):

    grouped_instances = {}
    index = 0
    while index < input_data.__len__():

        row = input_data.iloc[index]

        start_date = pd.Timestamp(year=int(row["Year"]), month=int(row["Month"]), day=int(row["Day"]), hour=int(row["Hour"]))
        end_date = start_date + pd.Timedelta(timeframe) - pd.Timedelta("1h")

        # Find instances within the grouping interval
        group_instances, count = GroupInstances(input_data[index:], start_date, end_date)
        index += count

        grouped_instances[start_date] = group_instances

    first_key = next(iter(grouped_instances))
    n_instances_batch = grouped_instances[first_key].__len__()

    new_columns = GenerateNewColumns(grouped_instances)
    new_dataset = pd.DataFrame(columns=new_columns)
    for index, group in grouped_instances.items():
        row = []
        for entry in group:
            for field in entry:
                row.append(field)

        if len(row) == len(new_columns):
            row_df = pd.DataFrame({new_columns[i]: [val] for i, val in enumerate(row)}, index=[0])
            new_dataset = pd.concat([new_dataset, row_df], ignore_index=True)

            #for n in range(len(row), len(new_columns)):
            #    row.append(np.NaN)

    return new_dataset, n_instances_batch


def predictConsumption(meteo_data, scheduling_data):
    # Predict the consumption taking into account the scheduling of the assets
    # The scheduling data must contain only the columns ['Year', 'Month', 'Day', 'Hour']

    print("*****************************************************************************************")

    data = pd.merge(meteo_data, scheduling_data, on=['Year', 'Month', 'Day', 'Hour'], how='inner')
    data_batches, n_per_batch = PrepareBatches(data, "1D")
    print(data_batches)

    consumption = cons_model.predict(data_batches)

    return consumption


def predictProduction(meteo_data, scheduling_data):
    # Predict the production taking into account the assets and its schedule
    # The scheduling data must contain only the columns ['Year', 'Month', 'Day', 'Hour']

    print("*****************************************************************************************")

    data = pd.merge(meteo_data, scheduling_data, on=['Year', 'Month', 'Day', 'Hour'], how='inner')
    data_batches, n_per_batch = PrepareBatches(data, "1D")

    production = prod_model.predict(data_batches)

    return production

