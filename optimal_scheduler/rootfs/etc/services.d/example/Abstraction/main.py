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

from homeassistant_api import Client

import OptimalScheduler as optimalscheduler

# URL for the Home Assistant API
# TODO: WORK WITH .secrets
ha_url = "http://192.168.1.192:8123"
bearer_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJiOWUzNjU4NWVkMzI0YzYxYWFlYTdhMmZiZTkyNGY0MCIsImlhdCI6MTcwNzMwMjM1OCwiZXhwIjoyMDIyNjYyMzU4fQ.d-brZLxCDdcUtuf5XpOjWjCBd-q4gPBgc18B7skr6z8"
ha = Client(ha_url, bearer_token) # Connect to Home Assistant instance

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
    message = str(sys.argv[1])

    print(message)
    print(ha.get_entity(entity_id="sun.sun").get_state()) #sensor.dht1_humidity






