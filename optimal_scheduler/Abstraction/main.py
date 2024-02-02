import multiprocessing
import threading
import time
import json

import OptimalScheduler as optimalscheduler
from Application import Application


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

        file = open("SavedOSConfigs/walqa.json")

        config = json.load(file)
        scheduler.deleteAssets()

        for asset_type in config:
            for asset_class in config[asset_type]:
                for asset in config[asset_type][asset_class].values():

                    scheduler.addAsset(asset_type, asset_class, asset)

        

if __name__ == "__main__":

    #event = threading.Event()
    scheduler = optimalscheduler.OptimalScheduler()

    importConfiguration()

    write_pipe = multiprocessing.Pipe()
    t2 = multiprocessing.Process(target=scheduler.startOptimizationNoPipe)#, args=(write_pipe,))
    t2.start()

    #app = Application(scheduler, event)

    #check_loop = threading.Thread(target=checkloop, args=(event,))
    #check_loop.start()

    #app.mainloop()






