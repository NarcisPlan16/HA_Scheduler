#!/usr/bin/python

import multiprocessing
import threading
import time
import json
import sys
import requests
import os
import subprocess
#from pathlib import Path

import OptimalScheduler as optimalscheduler

# URL for the Home Assistant API
# TODO: WORK WITH .secrets
ha_url = "http://192.168.0.117:8123"
bearer_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJkYjcxOTI3NmM2ZTA0YzU5YTZmM2YxZmFlOTUxZWM5OSIsImlhdCI6MTcxMDg2Nzc4NywiZXhwIjoyMDI2MjI3Nzg3fQ.72uuDLPBzDVVX7enOXmDlvI-eDcQxU_wPgAeHqw6eGs"

def backgroundSimulation(gui, os):

    read_pipe, write_pipe = multiprocessing.Pipe()

    t2 = multiprocessing.Process(target=os.startOptimization, args=(write_pipe,))
    t2.start()

    gui.updateProgress(read_pipe)


def checkloop(event: threading.Event):

    while True:

        event.wait()
        #backgroundSimulation(app, scheduler)

        time.sleep(2)

def importConfiguration():

    config = read_options()

    for asset_type in config:
        for asset_class in config[asset_type]:
            for asset in config[asset_type][asset_class].values():

                scheduler.addAsset(asset_type, asset_class, asset)


def read_options():
    pass


if __name__ == "__main__":

    #event = threading.Event()
    scheduler = optimalscheduler.OptimalScheduler()

    #importConfiguration()

    #write_pipe = multiprocessing.Pipe()
    #t2 = multiprocessing.Process(target=scheduler.startOptimizationNoPipe)#, args=(write_pipe,))
    #t2.start()

    #read_options()
    message = str(sys.argv[2])

    print(message)
    #print(ha.get_error_log()) #get_entity(entity_id="sensor.dht1_humidity")

headers = {
    "Authorization": f"Bearer {bearer_token}", #str(sys.argv[1]) for SUPERVISED_TOKEN
    "Content-Type": "application/json",
}

sensor_entity_id = "sensor.dht1_humidity"
# Make a GET request to retrieve the state of the sensor
response = requests.get(f"{ha_url}/api/states/{sensor_entity_id}", headers=headers) #http://supervisor/core/api/states/{sensor_entity_id} 

if response.status_code == 200:
    try:
        # Try to decode the JSON response
        sensor_state = response.text#.json() #["state"]
        print(f"The current state of {sensor_entity_id} is: {sensor_state}")
    except json.JSONDecodeError as e:
        # If response is not JSON, print the response content
        print("Response is not in JSON format:")
        print(response.text)
        print("JSONDecodeError:", e)
else:
    # Print error message if request was not successful
    print(f"Failed to retrieve state for sensor {sensor_entity_id}. Status code: {response.status_code}")

