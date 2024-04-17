import requests

ha_url = "http://192.168.0.110:8123"
bearer_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJlYzZhYjAxYTVkM2M0OGE3YjU0OGQ1NmYxNjQyNWQ2ZCIsImlhdCI6MTcxMzM1MDQxNSwiZXhwIjoyMDI4NzEwNDE1fQ.Eutl8pls09_KCIWESOv17gmIzu-RW32eazbHp2V4Wr0"

headers = {
    "Authorization": f"Bearer {bearer_token}",
    "Content-Type": "application/json",
}

entity = "sensor.solarnet_power_photovoltaics"
ini = "2023-01-01"
end = "2024-04-16"  # Year - month - Day
response = requests.get(f"{ha_url}/api/history/period/"+ini+"T00:00:00?end_time="+end+"T00:00:00&filter_entity_id="+entity, headers=headers)

if response.status_code == 200:
    data = response.json()

    # Extracting only the "state" variables
    states = []
    for entry in data:
        states.append((entry['last_updated'], entry['state']))

    print(states)

else:
    print(response.text)
