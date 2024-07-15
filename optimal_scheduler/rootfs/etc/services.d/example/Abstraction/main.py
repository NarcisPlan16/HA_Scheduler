#!/usr/bin/python

import multiprocessing
import threading
import time
import json
import sys
import requests
import os
import subprocess
import tomli
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
                print(f"[ERROR]: Consumer {consumer} not found in consumer list. Check the consumer ID")
                break

def checkGenerators(entity_ids):

    if str(sys.argv[3]): # the string is not empty

        generators = str(sys.argv[3]).split("\n") # Convert the inputed consumers string into an array of strings. They must be separated by enlines (\n)
        for generator in generators:
            if generator not in entity_ids:
                print(f"[ERROR]: Generator {generator} not found in generators list. Check the generator ID")
                break

def checkEnergySources(entity_ids):

    if str(sys.argv[4]): # the string is not empty

        esources = str(sys.argv[4]).split("\n") # Convert the inputed consumers string into an array. They must be separated by enlines (\n)
        for esource in esources:
            if esource not in entity_ids:
                print(f"[ERROR]: Energy Source {esource} not found in energy sources list. Check the energy source ID")
                break

def checkBuildingsConsumption(entity_ids): 

    if str(sys.argv[5]): # the string is not empty
        
        buildings_cons = str(sys.argv[5]).split("\n") # Convert the inputed consumers string into an array. They must be separated by enlines (\n)
        for building_cons in buildings_cons:
            if building_cons not in entity_ids:
                print(f"[ERROR]: Building consumption {building_cons} not found in buildings consumptions list. Check the building consumption ID")
                break

def checkBuildingsGeneration(entity_ids):

    if str(sys.argv[6]): # the string is not empty

        buildings_gen = str(sys.argv[6]).split("\n") # Convert the inputed consumers string into an array. They must be separated by enlines (\n)
        for building_gen in buildings_gen:
            if building_gen not in entity_ids:
                print(f"[ERROR]: Building generation {building_gen} not found in buildings generations list. Check the building generation ID")
                break

def pairSimulationFiles():

    result = {"Consumers": {}, "Generators": {}, "Energy Sources": {}}
    
    # Convert the inputed consumers, generators and energy sources strings into an array. They must be separated by enlines (\n)
    list_simu_dir = os.listdir("/share/config_optimal_scheduler/MySimulationCode") # Get the list of files on the config directory
    list_class_dir = os.listdir("/share/config_optimal_scheduler/MyClassesCode") # Get the list of files on the classes directory

    if str(sys.argv[2]): # String not empty

        entities = str(sys.argv[2]).split("\n") 
        for entity in entities:
            if list_simu_dir.__contains__("SIMU_"+entity+".py") and list_class_dir.__contains__(entity+".py") and list_simu_dir.__contains__("SIMU_"+entity+".toml"):
                result["Consumers"][entity] = {}
                result["Consumers"][entity]["Simulate"] = "/share/config_optimal_scheduler/MySimulationCode/SIMU_"+entity+".py"
                result["Consumers"][entity]["Class"] = "/share/config_optimal_scheduler/MyClassesCode/"+entity+".py"
                result["Consumers"][entity]["New_attributes"] = "/share/config_optimal_scheduler/MySimulationCode/SIMU_"+entity+".toml"
            else:
                print(f"[ERROR]: Simulation or class code not found for entity {entity}")

    if str(sys.argv[3]): # String not empty

        entities = str(sys.argv[3]).split("\n") 
        for entity in entities:
            if list_simu_dir.__contains__("SIMU_"+entity+".py") and list_class_dir.__contains__(entity+".py"):
                result["Generators"][entity] = {}
                result["Generators"][entity]["Simulate"] = "/share/config_optimal_scheduler/MySimulationCode/SIMU_"+entity+".py"
                result["Generators"][entity]["Class"] = "/share/config_optimal_scheduler/MyClassesCode/"+entity+".py"
                result["Generators"][entity]["New_attributes"] = "/share/config_optimal_scheduler/MySimulationCode/SIMU_"+entity+".toml"
            else:
                print(f"[ERROR]: Simulation or class code not found for entity {entity}")

    if str(sys.argv[4]): # String not empty

        entities = str(sys.argv[4]).split("\n") 
        for entity in entities:
            if list_simu_dir.__contains__("SIMU_"+entity+".py") and list_class_dir.__contains__(entity+".py"):
                result["Energy Sources"][entity] = {}            
                result["Energy Sources"][entity]["Simulate"] = "/share/config_optimal_scheduler/MySimulationCode/SIMU_"+entity+".py"
                result["Energy Sources"][entity]["Class"] = "/share/config_optimal_scheduler/MyClassesCode/"+entity+".py"
                result["Energy Sources"][entity]["New_attributes"] = "/share/config_optimal_scheduler/MySimulationCode/SIMU_"+entity+".toml"
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
            file_data = tomli.load(file)
            for key, value in file_data.items():
                data["attributes"][key] = value

        return data["attributes"]
    else:
        print(f"[ERROR] configure(): Entity {entity} not found")



def initEntities(entity_type, entities_list: dict, scheduler: optimalscheduler.OptimalScheduler):

    for entity, files in entities_list.items():
        asset_config = configure(entity, files) # entity name, entity simula and entity class
        scheduler.addAsset(entity_type, entity, asset_config)

def initBuildings(scheduler: optimalscheduler.OptimalScheduler):

    consumption = str(sys.argv[5]).split("\n")
    if consumption != ['']:
        for entity in consumption:
            scheduler.AddBuilding("Consumption", entity)

    generation = str(sys.argv[6]).split("\n") 
    if generation != ['']:
        for entity in generation:
            scheduler.AddBuilding("Generation", entity)

def startSimulation(paired_entities):

    scheduler = optimalscheduler.OptimalScheduler()
    initEntities("Consumers", paired_entities["Consumers"], scheduler)
    initEntities("Generators", paired_entities["Generators"], scheduler)
    initEntities("EnergySources", paired_entities["Energy Sources"], scheduler)
    initBuildings(scheduler)

    if scheduler.n_assets > 0:
        scheduler.startOptimizationNoPipe()
    else:
        print("+-----------------------------------------------------------------------------------------+")
        print("|                                                                                         |")
        print("|                No assets to be optimized. Maybe you forgot to add some?                 |")
        print("|                                                                                         |")
        print("+-----------------------------------------------------------------------------------------+")

if __name__ == "__main__":

    headers = {
        "Authorization": f"Bearer {bearer_token}", #str(sys.argv[1]) for SUPERVISED_TOKEN
        "Content-Type": "application/json",
    }

    # Make a GET request to retrieve the states
    response = requests.get(f"{ha_url}/api/states", headers=headers) #http://supervisor/core/api/states/{sensor_entity_id} 
    
    if response.status_code == 200:
        try:

            entity_ids = [entity["entity_id"] for entity in response.json()] # Extract only the entity ids

            checkConsumers(entity_ids)
            checkGenerators(entity_ids)
            checkEnergySources(entity_ids)
            checkBuildingsConsumption(entity_ids)
            checkBuildingsGeneration(entity_ids)

            paired_entities = pairSimulationFiles()
            startSimulation(paired_entities)

            time.sleep(20)

        except json.JSONDecodeError as e:
            # If response is not JSON, print the response content
            print("Response is not in JSON format:")
            print(response.text)
            print("JSONDecodeError:", e)
    else:
        # Print error message if request was not successful
        print(response.text)
