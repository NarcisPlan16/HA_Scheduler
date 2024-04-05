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

def pairSimulationFiles():

    result = {"Consumers": {}, "Generators": {}, "Energy Sources": {}}
    
    # Convert the inputed consumers, generators and energy sources strings into an array. They must be separated by enlines (\n)
    list_simu_dir = os.listdir("/config/OptimalScheduler/MySimulationCode") # Get the list of files on the config directory
    list_class_dir = os.listdir("/config/OptimalScheduler/MyClassesCode") # Get the list of files on the classes directory

    entities = str(sys.argv[2]).split("\n") 
    for entity in entities:
        if list_simu_dir.__contains__("SIMU_"+entity+".py") and list_class_dir.__contains__(entity+".py"):
            result["Consumers"][entity] = {}
            result["Consumers"][entity]["Simulate"] = "/config/OptimalScheduler/MySimulationCode/SIMU_"+entity+".py"
            result["Consumers"][entity]["Class"] = "/config/OptimalScheduler/MyClassesCode/"+entity+".py"

    entities = str(sys.argv[3]).split("\n") 
    for entity in entities:
        if list_simu_dir.__contains__("SIMU_"+entity+".py") and list_class_dir.__contains__(entity+".py"):
            result["Generators"][entity] = {}
            result["Generators"][entity]["Simulate"] = "/config/OptimalScheduler/MySimulationCode/SIMU_"+entity+".py"
            result["Generators"][entity]["Class"] = "/config/OptimalScheduler/MyClassesCode/"+entity+".py"

    entities = str(sys.argv[4]).split("\n") 
    for entity in entities:
        if list_simu_dir.__contains__("SIMU_"+entity+".py") and list_class_dir.__contains__(entity+".py"):
            result["Energy Sources"][entity] = {}            
            result["Energy Sources"][entity]["Simulate"] = "/config/OptimalScheduler/MySimulationCode/SIMU_"+entity+".py"
            result["Energy Sources"][entity]["Class"] = "/config/OptimalScheduler/MyClassesCode/"+entity+".py"

    return result

def configure(entity, files):
    
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Content-Type": "application/json",
    }

    response = requests.get(f"{ha_url}/api/states/{entity}", headers=headers)
    if response.status_code == 200:
        data = response.json()
        data["attributes"]["name"] = entity # add the name field into the config

        headers = {
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json",
        }
        body_sim = {
            "file_path": f"/config/OptimalScheduler/MySimulationCode/SIMU_{files["Simulate"]}"
        }
        body_class = {
            "file_path": f"/config/OptimalScheduler/MyClassesCode/{files["Class"]}"
        }

        response_simulate = requests.get(f"{ha_url}/api/states/file.path", headers=headers, data=body_sim)
        response_class = requests.get(f"{ha_url}/api/states/file.path", headers=headers, data=body_class)
        if response_simulate.status_code == 200 and response_class.status_code == 200:
            print(str(response_simulate.json()))
            data["attributes"]["Simulate"] = str(response_simulate.json())
            data["attributes"]["Class"] = str(response_class.json())
            print("main configure: " + data["attributes"]) # NO ENTRA A AQUI (dins l'if)....
            return data["attributes"]



def initEntities(entity_type, entities_list: dict, scheduler: optimalscheduler.OptimalScheduler):

    for entity, files in entities_list.items():
        asset_config = configure(entity, files) # entity name, entity simula and entity class
        print("main: " + asset_config)
        scheduler.addAsset(entity_type, entity, asset_config)

def startSimulation(paired_entities):

    scheduler = optimalscheduler.OptimalScheduler()
    consumers = initEntities("Consumers", paired_entities["Consumers"], scheduler)
    generators = initEntities("Generators", paired_entities["Generators"], scheduler)
    energy_sources = initEntities("EnergySources", paired_entities["Energy Sources"], scheduler)
    scheduler.startOptimizationNoPipe()


if __name__ == "__main__":

    headers = {
        "Authorization": f"Bearer {bearer_token}", #str(sys.argv[1]) for SUPERVISED_TOKEN
        "Content-Type": "application/json",
    }

    # Make a GET request to retrieve the state of the sensor
    response = requests.get(f"{ha_url}/api/states", headers=headers) #http://supervisor/core/api/states/{sensor_entity_id} 
    
    if response.status_code == 200:
        try:

            entity_ids = [entity["entity_id"] for entity in response.json()] # Extract only the entity ids
            #for entity in entity_ids:
            #    print(entity)

            checkConsumers(entity_ids)
            checkGenerators(entity_ids)
            checkEnergySources(entity_ids)
            checkBuilding(entity_ids)

            print("[DEBUG]: All entities found!")

            paired_entities = pairSimulationFiles()
            if paired_entities.__len__ == 0 and entity_ids.__len__ > 0:
                print("[DEBUG]: Some files for the simulation not found")

            #print(paired_entities)

            startSimulation(paired_entities)

        except json.JSONDecodeError as e:
            # If response is not JSON, print the response content
            print("Response is not in JSON format:")
            print(response.text)
            print("JSONDecodeError:", e)
    else:
        # Print error message if request was not successful
        print(f"Failed to retrieve state for sensor {sensor_entity_id}. Status code: {response.status_code}")

