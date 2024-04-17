import requests
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

ha_url = "http://192.168.0.110:8123"
bearer_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJlYzZhYjAxYTVkM2M0OGE3YjU0OGQ1NmYxNjQyNWQ2ZCIsImlhdCI6MTcxMzM1MDQxNSwiZXhwIjoyMDI4NzEwNDE1fQ.Eutl8pls09_KCIWESOv17gmIzu-RW32eazbHp2V4Wr0"

headers = {
    "Authorization": f"Bearer {bearer_token}",
    "Content-Type": "application/json",
}

entity = "sensor.sonnenbatterie_79259_meter_production_4_1_w_total"
ini = "2023-01-01"
end = "2024-04-16"  # Year - month - Day
response = requests.get(f"{ha_url}/api/history/period/"+ini+"T00:00:00?end_time="+end+"T00:00:00&filter_entity_id="+entity, headers=headers)

if response.status_code == 200:

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

    print(data)
    data.plot()
    plt.show()

    print("Preprocessing done")

    # TODO: Start forecasting. En llorenç m'ha recomanat passar-li "batches" de 1 dia, 1 setmana o 1 mes
    model = RandomForestRegressor(n_estimators=4, max_depth=8, random_state=0, n_jobs=4)
    model.fit(data.index, data['state'])

    now = datetime(datetime.today().year, datetime.today().month, datetime.today().day, datetime.today().hour)
    print(now)
    print(model.predict(now))

else:
    print(response.text)
