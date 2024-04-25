import math

import numpy as np
import requests
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns

ha_url = "http://192.168.0.110:8123"
bearer_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJlYzZhYjAxYTVkM2M0OGE3YjU0OGQ1NmYxNjQyNWQ2ZCIsImlhdCI6MTcxMzM1MDQxNSwiZXhwIjoyMDI4NzEwNDE1fQ.Eutl8pls09_KCIWESOv17gmIzu-RW32eazbHp2V4Wr0"

headers = {
    "Authorization": f"Bearer {bearer_token}",
    "Content-Type": "application/json",
}


def GroupInstances(input_data: pd.DataFrame, start_date, end_date):

    res = []

    print(input_data.__len__())

    date = start_date
    index = 0
    while start_date <= date <= end_date and index < input_data.__len__():

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
        end_date = start_date + pd.Timedelta(timeframe)

        # Find instances within the grouping interval
        group_instances, count = GroupInstances(input_data[index:], start_date, end_date)
        index += count

        grouped_instances[start_date] = group_instances

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

    return new_dataset


def SeparateXY(dataframe: pd.DataFrame):

    Y_rows = [col for col in dataframe.columns if "state" in col]
    X_data = dataframe.drop(columns=Y_rows)
    Y_data = dataframe[Y_rows]

    return X_data, Y_data




ini = "2022-01-01"
end = "2024-04-16"  # Year - month - Day
request_to_api = False
if request_to_api:

    entity = "sensor.sonnenbatterie_79259_meter_production_4_1_w_total"
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

    data.to_json('PVProduction.json', orient='split', compression='infer', index='true')

else:
    data = pd.read_json('PVProduction.json', orient='split', compression='infer')

#print(data)
#data[0:72].plot()
#plt.show()

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
print("Preparing data")

# TODO: Start forecasting. En llorenç m'ha recomanat passar-li "batches" de 1 dia, 1 setmana o 1 mes. Per fer això,
#       m'ha recomanat que afegeixi atributs, per tant per una instància X tindré
#       X_dia1_hora:18h, X_dia1_hora19h, ..., X_dia2_hora11h

#data = PrepareBatches(data, "7D")

print("Data is ready, starting training and model fit")

sns.set_theme(style="white")
corr = data.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))  # Generate a mask for the upper triangle

f, ax = plt.subplots(figsize=(14, 12))
cmap = sns.diverging_palette(230, 20, as_cmap=True)  # Generate a custom diverging colormap

# Draw the heatmap with the mask and correct aspect ratio
heatmap = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()

fig = heatmap.get_figure()
fig.savefig("correlation_matrix.png")

train_size = math.floor(len(data)*0.8)
data_X, data_y = SeparateXY(data)
X_train = data_X[0:train_size]
y_train = data_y[0:train_size]
X_test = data_X[train_size:]
y_test = data_y[train_size:]

print("Dataset instances: " + str(data.shape[0]))
print("Dataset attributes: " + str(data.shape[1]))
model = RandomForestRegressor(n_estimators=int(data.shape[0]*0.4), max_depth=int(data.shape[1]*0.5), random_state=0, n_jobs=4, verbose=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("MSE: ", mse)
mape = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE: ", mape)

#timestamps = pd.to_datetime(X_test['Year', 'Month', 'Day', 'Hour'], format='%Y-%m-%d %H:%M:%S')
#plt.figure(figsize=(10, 6))
#x = [i for i in range(0, 76)]
#plt.scatter(x, y_test, color='blue', label='y_test', marker='.')
#plt.scatter(x, y_pred, color='orange', label='y_pred', marker='.')
#plt.xlabel('X_test')
#plt.ylabel('Values')
#plt.title('Predicted production (Kwh)')
#plt.legend()
#plt.show()
