import multiprocessing
import threading
import time
import json
import requests

import OptimalScheduler as optimalscheduler

# URL for the Home Assistant API
# TODO: WORK WITH .secrets
api_url = "http://192.168.1.192:8123/api/"
access_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJiOWUzNjU4NWVkMzI0YzYxYWFlYTdhMmZiZTkyNGY0MCIsImlhdCI6MTcwNzMwMjM1OCwiZXhwIjoyMDIyNjYyMzU4fQ.d-brZLxCDdcUtuf5XpOjWjCBd-q4gPBgc18B7skr6z8"

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
        #"Authorization": f"Bearer {access_token}"
        #"Bearer Token": access_token,
        "Content-Type": "application/json"
    }

    # Send GET request to fetch options
    response = requests.get(api_url+"/api/config/", headers=headers)
    print(response)
    
    if response.status_code == 200:
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

    read_options()








