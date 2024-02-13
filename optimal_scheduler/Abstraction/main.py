import multiprocessing
import threading
import time
import json
import requests
import logging
import os
import yaml
from pathlib import Path

import OptimalScheduler as optimalscheduler

# URL for the Home Assistant API
# TODO: WORK WITH .secrets
api_url = "http://192.168.0.117:8123/api/"
bearer_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJiOWUzNjU4NWVkMzI0YzYxYWFlYTdhMmZiZTkyNGY0MCIsImlhdCI6MTcwNzMwMjM1OCwiZXhwIjoyMDIyNjYyMzU4fQ.d-brZLxCDdcUtuf5XpOjWjCBd-q4gPBgc18B7skr6z8"

def backgroundSimulation(gui, os):

    read_pipe, write_pipe = multiprocessing.Pipe()

    t2 = multiprocessing.Process(target=os.startOptimization, args=(write_pipe,))
    t2.start()

    gui.updateProgress(read_pipe)


def checkloop(event: threading.Event):

    while True:

        event.wait()
        backgroundSimulation(app, scheduler)

        time.sleep(2)

def importConfiguration():

        #current_dir = os.getcwd()
        #save_dir = os.path.join(current_dir, "SavedOSConfigs")

        #files = [('Json Files', '*.json')]
        #file = fd.askopenfile(filetypes=files, initialdir=save_dir, defaultextension="json")

        file = open("/Abstraction/SavedOSConfigs/walqa.json")

        config = json.load(file)
        scheduler.deleteAssets()

        for asset_type in config:
            for asset_class in config[asset_type]:
                for asset in config[asset_type][asset_class].values():

                    scheduler.addAsset(asset_type, asset_class, asset)


def read_options():
    
    # Headers for API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['SUPERVISOR_TOKEN']}"
    }

    #OPTIONS_PATH = os.getenv('OPTIONS_PATH', default="/data/options.json")
    #options_json = Path(OPTIONS_PATH)

    # Read options info
    #if options_json.exists():
    #    with options_json.open('r') as data:
    #        options = json.load(data)
        #print(options)
    #else:
    #   print("options.json does not exist")

    # Send GET request to fetch options
    response = requests.get(url="http://supervisor/"+"addons/self", headers=headers) #"http://supervisor/"
    print(response.text)

    if response.status_code == 200:
        print(response.json())
        options = response.json()["data"]

        # Access the options
        message = options.get("message", "Default message")
        toggle_something = options.get("Toggle something", False)
        toggle_something_else = options.get("Toggle something else", False)

        # Now you can use these options as needed
        print("Message:", message)
        print("Toggle something:", toggle_something)
        print("Toggle something else:", toggle_something_else)
    else:
        print("Error fetching options:", response.text)


if __name__ == "__main__":

    #event = threading.Event()
    scheduler = optimalscheduler.OptimalScheduler()

    importConfiguration()

    write_pipe = multiprocessing.Pipe()
    t2 = multiprocessing.Process(target=scheduler.startOptimizationNoPipe)#, args=(write_pipe,))
    #t2.start()

    #read_options()

    file = open("/addon_configs/local_optimal_scheduler/optimal_scheduler.yaml")
    config = yaml.safe_load(file)
    print(config)






