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
from flask import Flask


# URL for the Home Assistant API
# TODO: WORK WITH .secrets
ha_url = "http://192.168.1.192:8123"
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


def checkConsumers(entity_ids):

    if str(sys.argv[2]): # the string is not empty

        consumers = str(sys.argv[2]).split("\n") # Convert the inputed consumers string into an array. They must be separated by enlines (\n)
        for consumer in consumers:
            if consumer not in entity_ids:
                print(f"[ERROR]: Consumer {consumer} not found in consumer ist. Check the consumer ID")
                break

def checkGenerators(entity_ids):

    if str(sys.argv[3]): # the string is not empty

        generators = str(sys.argv[3]).split("\n") # Convert the inputed consumers string into an array of strings. They must be separated by enlines (\n)
        for generator in generators:
            if generator not in entity_ids:
                print(f"[ERROR]: Generator {generator} not found in generators ist. Check the generator ID")
                break

def checkEnergySources(entity_ids):

    if str(sys.argv[4]): # the string is not empty

        esources = str(sys.argv[4]).split("\n") # Convert the inputed consumers string into an array. They must be separated by enlines (\n)
        for esource in esources:
            if esource not in entity_ids:
                print(f"[ERROR]: Energy Source {esource} not found in energy sources ist. Check the energy source ID")
                break

def checkBuilding(entity_ids):
    pass


if __name__ == "__main__":

    app = Flask(__name__)

    @app.route('/')
    def hello_world():
        return 'Hello, World! This is my addon website.'

    #event = threading.Event()
    scheduler = optimalscheduler.OptimalScheduler()

    #write_pipe = multiprocessing.Pipe()
    #t2 = multiprocessing.Process(target=scheduler.startOptimizationNoPipe)#, args=(write_pipe,))
    #t2.start()

    headers = {
        "Authorization": f"Bearer {bearer_token}", #str(sys.argv[1]) for SUPERVISED_TOKEN
        "Content-Type": "application/json",
    }

    sensor_entity_id = "sensor.dht1_humidity"
    # Make a GET request to retrieve the state of the sensor
    response = requests.get(f"{ha_url}/api/states", headers=headers) #http://supervisor/core/api/states/{sensor_entity_id} 
    
    if response.status_code == 200:
        try:

            entity_ids = [entity["entity_id"] for entity in response.json()] # Extract only the entity ids
            #for entity in entity_ids:
            #    print(entity)

            print(str(sys.argv[2]))
            checkConsumers(entity_ids)
            checkGenerators(entity_ids)
            checkEnergySources(entity_ids)
            checkBuilding(entity_ids)

            print("[DEBUG]: All entities found!")

        except json.JSONDecodeError as e:
            # If response is not JSON, print the response content
            print("Response is not in JSON format:")
            print(response.text)
            print("JSONDecodeError:", e)
    else:
        # Print error message if request was not successful
        print(f"Failed to retrieve state for sensor {sensor_entity_id}. Status code: {response.status_code}")

