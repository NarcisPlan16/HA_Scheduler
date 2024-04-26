import math

import numpy as np
import requests
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import cross_val_score, cross_val_predict
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


def SeparateXY(dataframe: pd.DataFrame):

    Y_rows = [col for col in dataframe.columns if "state" in col]
    X_data = dataframe.drop(columns=Y_rows)
    Y_data = dataframe[Y_rows]

    return X_data, Y_data


def CalcCorrMatrix(dataset: pd.DataFrame, display: bool):

    corr = dataset.corr()
    if display:
        sns.set_theme(style="white")
        mask = np.triu(np.ones_like(corr, dtype=bool))  # Generate a mask for the upper triangle

        f, ax = plt.subplots(figsize=(14, 12))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)  # Generate a custom diverging colormap

        # Draw the heatmap with the mask and correct aspect ratio
        heatmap = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                              square=True, linewidths=.5, cbar_kws={"shrink": .5})

        plt.show()

        fig = heatmap.get_figure()
        fig.savefig("correlation_matrix.png")

    return corr


def CleanByCorrelation(corr_mat, dataset: pd.DataFrame):

    columns_to_remove = corr_mat.index[((corr_mat["state"] < 0.2) &
                                        (corr_mat["state"] > -0.2)) & (corr_mat.index != "state")]

    selected_columns = []
    for col in dataset.columns:
        if not any(exclude_str in col for exclude_str in columns_to_remove):
            selected_columns.append(col)

    df_filtered = dataset[selected_columns]

    return df_filtered


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
    data = data.drop(columns='Timestamp', axis=1)

    data.to_json('Data_Plus_MeteoForecast.json', orient='split', compression='infer', index=True)
else:
    data = pd.read_json('Data_Plus_MeteoForecast.json', orient='split', compression='infer')

print("Preprocessing done")
print("Preparing data")

data_batches, n_per_batch = PrepareBatches(data, "1D")
corr_matrix = CalcCorrMatrix(data, False)
data = CleanByCorrelation(corr_matrix, data_batches)

print("Data is ready, starting training and model fit")

train_size = math.floor(len(data)*0.8)
data_X, data_y = SeparateXY(data)
X_train = data_X[0:train_size]
y_train = data_y[0:train_size]
X_test = data_X[train_size:]
y_test = data_y[train_size:]

total_hours = n_per_batch * data.shape[0]
print("Dataset instances: " + str(data.shape[0]))
print("Dataset attributes: " + str(data.shape[1]))
print("Total hour instances: " + str(total_hours))
print(X_train.head())
model = RandomForestRegressor(n_estimators=int(total_hours*0.2), max_depth=int(X_train.shape[1]*0.7), random_state=0, n_jobs=-1, verbose=True)
print(model)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("MSE: ", mse)
mape = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE: ", mape)
r2 = r2_score(y_test, y_pred)
print("R2 score: ", r2)

print("----------------------Now trying with cross validation----------------------")
y_pred = cross_val_predict(model, X_train, y_train, cv=5, n_jobs=-1) # 328
mse = mean_squared_error(y_test, y_pred)  # TODO: y_pred te size 328 x n_atributs que és 5*82 (82 és el nombre d'instàncies d'y_test)
print("MSE: ", mse)
mape = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE: ", mape)
r2 = r2_score(y_test, y_pred)
print("R2 score: ", r2)

#timestamps = pd.to_datetime(X_test['Year', 'Month', 'Day', 'Hour'], format='%Y-%m-%d %H:%M:%S')
plt.figure(figsize=(10, 6))
x = [i for i in range(0, y_test[0:6].size)]
plt.scatter(x, y_test[0:6], color='blue', label='y_test', marker='.')
plt.scatter(x, y_pred[0:6], color='orange', label='y_pred', marker='.')
plt.xlabel('X_test')
plt.ylabel('PV Production')
plt.title('Predicted production (Kwh)')
plt.legend()
plt.show()
