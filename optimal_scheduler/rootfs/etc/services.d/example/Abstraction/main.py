#!/usr/bin/python

import multiprocessing
import threading
import time
import json
import sys
import requests
import os
import subprocess
import tomllib
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

    if str(sys.argv[2]): # String not empty

        entities = str(sys.argv[2]).split("\n") 
        for entity in entities:
            if list_simu_dir.__contains__("SIMU_"+entity+".py") and list_class_dir.__contains__(entity+".py") and list_simu_dir.__contains__("SIMU_"+entity+".toml"):
                result["Consumers"][entity] = {}
                result["Consumers"][entity]["Simulate"] = "/config/OptimalScheduler/MySimulationCode/SIMU_"+entity+".py"
                result["Consumers"][entity]["Class"] = "/config/OptimalScheduler/MyClassesCode/"+entity+".py"
                result["Consumers"][entity]["New_attributes"] = "/config/OptimalScheduler/MySimulationCode/SIMU_"+entity+".toml"
            else:
                print(f"[ERROR]: Simulation or class code not found for entity {entity}")

    if str(sys.argv[3]): # String not empty

        entities = str(sys.argv[3]).split("\n") 
        for entity in entities:
            if list_simu_dir.__contains__("SIMU_"+entity+".py") and list_class_dir.__contains__(entity+".py"):
                result["Generators"][entity] = {}
                result["Generators"][entity]["Simulate"] = "/config/OptimalScheduler/MySimulationCode/SIMU_"+entity+".py"
                result["Generators"][entity]["Class"] = "/config/OptimalScheduler/MyClassesCode/"+entity+".py"
                result["Generators"][entity]["New_attributes"] = "/config/OptimalScheduler/MySimulationCode/SIMU_"+entity+".toml"
            else:
                print(f"[ERROR]: Simulation or class code not found for entity {entity}")

    if str(sys.argv[4]): # String not empty

        entities = str(sys.argv[4]).split("\n") 
        for entity in entities:
            if list_simu_dir.__contains__("SIMU_"+entity+".py") and list_class_dir.__contains__(entity+".py"):
                result["Energy Sources"][entity] = {}            
                result["Energy Sources"][entity]["Simulate"] = "/config/OptimalScheduler/MySimulationCode/SIMU_"+entity+".py"
                result["Energy Sources"][entity]["Class"] = "/config/OptimalScheduler/MyClassesCode/"+entity+".py"
                result["Energy Sources"][entity]["New_attributes"] = "/config/OptimalScheduler/MySimulationCode/SIMU_"+entity+".toml"
            else:
                print(f"[ERROR]: Simulation or class code not found for entity {entity}")

    return result

def configure(entity: str, files):
 
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Content-Type": "application/json",
    }

    response = requests.get(f"{ha_url}/api/states/{entity}", headers=headers)
    if response.status_code == 200:

        data = response.json()
        data["attributes"]["name"] = entity # add the name field into the config

        with open(files["Simulate"], 'r') as file:
            data["attributes"]["Simulate"] = file.read()
        with open(files["Class"], 'r') as file:
            data["attributes"]["Class"] = file.read()
        with open(files["New_attributes"], 'rb') as file:
            file_data = tomllib.load(file)
            for key, value in file_data.items():
                data["attributes"][key] = value
                
        return data["attributes"]
    else:
        print(f"[ERROR] configure(): Entity {entity} not found")



def initEntities(entity_type, entities_list: dict, scheduler: optimalscheduler.OptimalScheduler):

    for entity, files in entities_list.items():
        asset_config = configure(entity, files) # entity name, entity simula and entity class
        scheduler.addAsset(entity_type, entity, asset_config)

def startSimulation(paired_entities):

    scheduler = optimalscheduler.OptimalScheduler()
    initEntities("Consumers", paired_entities["Consumers"], scheduler)
    initEntities("Generators", paired_entities["Generators"], scheduler)
    initEntities("EnergySources", paired_entities["Energy Sources"], scheduler)
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

            paired_entities = pairSimulationFiles()
            startSimulation(paired_entities)

        except json.JSONDecodeError as e:
            # If response is not JSON, print the response content
            print("Response is not in JSON format:")
            print(response.text)
            print("JSONDecodeError:", e)
    else:
        # Print error message if request was not successful
        print(response.text)
