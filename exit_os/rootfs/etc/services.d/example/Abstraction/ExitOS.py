# Exit OS "class". Main program that runs the optimization algorithm. It is responsible for the optimization of the assets.

import math
import time
import json
import psutil
import copy
import os
import requests
import joblib
import logging
import sys
import ForecastersManager

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import cProfile # Uncomment for debug
#import pyswarms as ps
#import pyswarms.backend.topology as ps_tp
from Solution import Solution
from AbsConsumer import AbsConsumer
from AbsGenerator import AbsGenerator
from AbsEnergySource import AbsEnergySource
from Asset_types.Consumers.HidrogenStation.HidrogenStation import HidrogenStation
from Configurator import Configurator
#from scipy.optimize import differential_evolution, dual_annealing, direct, brute, Bounds
from geneticalgorithm.geneticalgorithm import geneticalgorithm
from datetime import datetime, timedelta

# Setup the connection with the Home Assistant API
ha_url = "http://192.168.0.117:8123"
bearer_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJmZThlNTgyNDBhYTA0M2UwOTYyMmRmZWJlMTc5MDc0YyIsImlhdCI6MTcxOTMwNDY4NiwiZXhwIjoyMDM0NjY0Njg2fQ.j8euYQxDWMkJJqHNpTXUBE1rrhpOm1Vr-WcY3fdt8q0"
# Note: If for some reason, we are getting a code 401 (Unauthorized) and previously we didn0t, try creating a new bearer token

# Http request headers
headers = {
    "Authorization": f"Bearer {bearer_token}",
    "Content-Type": "application/json",
}

# Logging setup to print log messages to the console
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Example usage
#logger.info("AAAAAAAAAAAAAAAAAA")

# Main class ExitOS
class ExitOS:
    """
    Class responsible for optimizing energy assets using different forecasting methods and optimization algorithms.

    Attributes
    -----------
    maxiter : int
        Maximum number of iterations for the optimization model.
    hores_simular : int
        Number of hours to simulate for the optimization process.
    electricity_prices : list
        Hourly electricity prices for buying.
    electricity_sell_prices : list
        Hourly electricity sell prices.
    latitude : float
        Latitude used to fetch meteorological data.
    longitude : float
        Longitude used to fetch meteorological data.
    meteo_data : DataFrame
        Meteorological data predictions for the simulation period.
    kwargs_for_simulating : dict
        Arguments passed to the simulation function, shared between assets for interaction during simulation.
    """

    def __init__(self):
        # Initialization of key attributes
        self.connections = []  # List to store network connections
        self.progress = []  # Track optimization progress (cost at each iteration)
        self.debug = True  # Debug mode flag
        self.console_debug = False  # Flag to debug using console

        self.maxiter = 30  # Maximum number of iterations for optimization

        self.hores_simular = 24  # Hours to simulate for the optimization process
        self.hidrogen_price = 1.6  # Price of hydrogen, relevant for optimization

        self.consumer_invalid_solutions = 0  # Counter for invalid consumer solutions (for debugging)
        self.generator_invalid_solutions = 0  # Counter for invalid generator solutions (for debugging)

        # Retrieve electricity buying prices from external source
        self.electricity_prices = self.__obtainElectricityPrices()

        # Fixed electricity sell prices (dummy data)
        self.electricity_sell_prices = [0.054] * 24

        # Location data to fetch meteorological data
        self.latitude = "41.963138"
        self.longitude = "2.831640"

        # Fetch meteorological data based on latitude and longitude
        self.meteo_data = ForecastersManager.obtainMeteoData(self.latitude, self.longitude)

        # Initialize empty lists for forecasts
        self.electrcity_production_forecast = []
        self.electricity_consumption_forecast = []

        # Initialize solutions (initial and final)
        self.solucio_run = Solution()
        self.solucio_final = Solution()

        # Asset details for consumers, generators, and energy sources
        self.n_assets = 0
        self.assets = {
            'Buildings': {'Consumption': [], 'Generation': []}, 
            'Consumers': {}, 
            'EnergySources': {}, 
            'Generators': {}
        } # TODO: Improve to create an entry for each folder in Asset_types automatically

        # Key arguments for simulation that allows communication between different assets
        self.kwargs_for_simulating = {}

    def __obtainElectricityPrices(self):
        """
        Fetches the hourly electricity prices for buying from the OMIE API.
        If the next day's data is unavaliable, today's data is fetched.

        Returns
        -----------
        hourly_prices : list
            List of hourly electricity prices for buying.
        """
        # Calculate the date for tomorrow
        tomorrow = datetime.today() + timedelta(1)
        tomorrow_str = tomorrow.strftime('%Y%m%d')

        # Fetch electricity price data from OMIE
        url = f"https://www.omie.es/es/file-download?parents%5B0%5D=marginalpdbc&filename=marginalpdbc_{tomorrow_str}.1"
        response = requests.get(url)

        # If tomorrow's prices are unavailable, fallback to today's data
        if response.status_code != 200:
            today = datetime.today().strftime('%Y%m%d')
            url = f"https://www.omie.es/es/file-download?parents%5B0%5D=marginalpdbc&filename=marginalpdbc_{today}.1"
            response = requests.get(url)

        # Save the retrieved data into a CSV file
        with open("omie_price_pred.csv", 'wb') as f:
            f.write(response.content)

        # Parse the CSV to extract hourly prices
        hourly_prices = []
        with open('omie_price_pred.csv', 'r') as file:
            for line in file.readlines()[1:-1]:
                components = line.strip().split(';')
                components.pop(-1)  # Remove trailing blank entry
                hourly_price = float(components[-1])
                hourly_prices.append(hourly_price)

        return hourly_prices
    
    def __preparePredData(self, type: str):
        """
        Prepares data for the prediction model for consumption or generation.

        Parameters
        ----------
        type : str
            The type of data to prepare: 'Consumption' or 'Generation'.

        Returns
        -------
        DataFrame
            A DataFrame containing the timestamp and state for the past 24 hours and future prediction data.
        """
        
        dictionary = {'Timestamp': [], 'state': []}
        res = pd.DataFrame(dictionary)

        today = datetime.today().replace(hour=0, minute=0, second=0)
        tomorrow = datetime.today() + timedelta(days=1)
        start_of_tomorrow = datetime(tomorrow.year, tomorrow.month, tomorrow.day)

        today_str =  (today + pd.Timedelta(hours=1)).strftime('%Y-%m-%d')
        tomorrow_str = tomorrow.strftime('%Y-%m-%d')

        # Iterate through each building type in the solution (e.g., Consumption, Generation)
        for building_type in self.solucio_run.buildings[type]:
            try:
                # Fetch building data from Home Assistant API
                response = requests.get(
                    f"{ha_url}/api/history/period/" + today_str + "T00:00:00?end_time=" + tomorrow_str + "T00:00:00&filter_entity_id=" + building_type,
                    headers=headers
                )
                response_data = response.json()[0]
                data = pd.DataFrame.from_dict(response_data)

                # Fill missing data if less than 24 hours of past data is available
                if len(data['state']) < 24:
                    print("[WARNING]: No data found for previous 24h. Using mean value as fallback")
                    mean = 0  # Placeholder for future improvement
                    for i in range(0, 24):
                        date = pd.to_datetime(today + timedelta(hours=i)).strftime('%Y-%m-%d %H:%M:%S')
                        res.loc[len(res.index)] = [date, mean]
                else:
                    for i in range(0, 24):
                        date = pd.to_datetime(today + timedelta(hours=i)).strftime('%Y-%m-%d %H:%M:%S')
                        res.loc[len(res.index)] = [date, data['state'][i]]

                # Add prediction data for the next 24 hours
                for i in range(0, 24):
                    date = pd.to_datetime(start_of_tomorrow + timedelta(hours=i)).strftime('%Y-%m-%d %H:%M:%S')
                    res.loc[len(res.index)] = [date, 0]

            except KeyError:
                print(f"[ERROR]: Couldn't get state data for {building_type}")

        return res
    
    def __calcMissingValueFiller(self, today, building_type):
        """
        Calculates the fallback value for missing data by averaging data from the last 7 days.

        Parameters
        ----------
        today : datetime
            Today's date and time.
        building_type : str
            Type of building asset to calculate missing data for.

        Returns
        -------
        float
            Mean value used to fill missing data.
        """
        found = False
        i = 0
        data = pd.DataFrame()
        mean = 0
        while i < 7 and not found:

            ini_str = today.strftime('%Y-%m-%d')
            end_str = (today - timedelta(days=i)).strftime('%Y-%m-%d')

            response = requests.get(
                        f"{ha_url}/api/history/period/" + ini_str + "T00:00:00?end_time=" + end_str + "T00:00:00&filter_entity_id=" + building_type,
                        headers=headers)
                                
            response_data = response.json()[0]
            data = data.from_dict(response_data)

            if not len(data['state']) < 24: 
                found = False

        if not found:
            print(f"[ERROR]: No recent data found for asset {building_type} on the previous 7 days")
        else:
            mean = data['state'].mean()
            
        return mean

    def __optimize(self):

        print("--------------------------RUNNING COST OPTIMIZATION ALGORITHM--------------------------")

        """
        Funcio que engega la optimitzacio corresponent. i retorna la millor comanda que s'ha trobat per retornar a l'ESB 
        """

        # DE
        #resultat = self.__runDEModel(self.costDE)

        # PSO
        #resultat = self.__runPSOModel(self.costPSO)

        # SA
        #resultat = self.__runSAModel(self.costSA)

        # DIRECT
        #resultat = self.__runDIRECTModel(self.costSA)

        # GA
        cons_data = self.__preparePredData('Consumption')
        self.building_consumption = ForecastersManager.predictConsumption(self.meteo_data, cons_data)

        prod_data = self.__preparePredData('Generation')
        self.building_production = ForecastersManager.predictProduction(self.meteo_data, prod_data)

        model = self.__initializeGAModel(len(self.varbound), self.costDE, self.varbound)
        resultat = self.__runGAModel(model)

        # Retornem la configuracio de les variables del model
        return resultat

    def __runDIRECTModel(self, function):

        """
        lb = []
        ub = []
        for element in self.varbound:
            lb.append(element[0])
            ub.append(element[1])

        bounds = Bounds(lb, ub)

        res = direct(func=function,
                     bounds=bounds,
                     eps=0.01,
                     maxiter=500,
                     locally_biased=False)

        """

        return #res

    def __updateDIRECTStep(self, cost):

        print("COST ACTUAL: ", cost)

    def __runSAModel(self, function):

        """
        res = dual_annealing(func=function,
                             bounds=self.varbound,
                             maxiter=500,
                             no_local_search=True,
                             callback=self.__updateSAStep)

        """
        return #res

    def __updateSAStep(self, config, cost, context):

        print("MINIM TROBAT: ", cost)
        print("CONTEXT: ", context)


    def __runPSOModel(self, function):

        # Set-up hyperparameters and topology
        # One proven good parameter set: {'c1': 0.75, 'c2': 0.1, 'w': 0.99}
        options = {'c1': 1.7, 'c2': 0.4, 'w': 0.75}
        lb = []
        ub = []
        for element in self.varbound:
            lb.append(element[0])
            ub.append(element[1])

        # Call instance of GlobalBestPSO
        #optimizer = ps.single.GlobalBestPSO(n_particles=800,
        #                                    dimensions=len(self.varbound),
        #                                    bounds=[lb, ub],
        #                                    options=options)

        # Perform optimization
        #cost, config = optimizer.optimize(function,
        #                                  iters=500,
        #                                  n_processes=6,
        #                                  verbose=True)

        #resultat = self.Result(config)
        #cost = function([config])
        #self.solucio_final = copy.deepcopy(self.solucio_run)

        return #resultat

    def __runGAModel(self, model):

        # engeguem la simulacio
        time_inici = time.time()
        model.run()
        time_fi = time.time()

        print("\nTemps tardat: " + str(time_fi - time_inici))
        print("\nNumero d'assets ok: ", self.solucio_final.numero_assets_ok)

        i = 0
        while model.best_function > 999999 or self.solucio_run.numero_assets_ok != len(self.solucio_run.consumers) + len(self.solucio_run.energy_sources):

            print(' Solucio no trobada, recalculant!!')
            time_inici = time.time()
            model.run()
            time_fi = time.time()

            print("\nTemps tardat: " + str(time_fi - time_inici))
            print("\nNumero de consumers ok: ", self.solucio_final.numero_assets_ok)

            i = i + 1
            if i == 5:
                return ' Error - Solucio no trobada'

        # retornem la millor solucio trobada (configuracio de vbounds)
        return model.best_variable

    def __initializeGAModel(self, n_particules, funcio, varbound):

        # parametres de l'algorisme
        algorithm_param = {'max_num_iteration': 4000,
                           'population_size': 100,
                           'mutation_probability': 0.15,
                           'elit_ratio': 0.01,
                           'crossover_probability': 0.5,
                           'parents_portion': 0.3,
                           'crossover_type': 'uniform',
                           'max_iteration_without_improv': 900}  # o none per forçar que faci totes les iteracions

        # creem el GA solver
        model = geneticalgorithm(function=funcio,
                                 dimension=n_particules,
                                 variable_type='int',
                                 variable_boundaries=varbound,
                                 algorithm_parameters=algorithm_param)

        # desactivem el progress var si no estem en debug
        model.progress_bar = self.debug

        # desactivem el plot si no estem en debug
        model.convergence_curve = self.debug

        return model

    def __updateDEStep(self, bounds, convergence):
        # funcio per guardar els valors del millor cost de cada step i després poder mostrar-los

        t_ini = time.time()

        self.solucio_final.model_variables = bounds
        cost = self.costDE(bounds)

        self.solucio_final = copy.deepcopy(self.solucio_run)

        t_fi = time.time()

        self.progress.append(cost - self.solucio_run.cost_aproximacio)

        for connection in self.connections:
            connection.send([len(self.progress), self.maxiter])

        print("COST APROXIMACIO: ", self.solucio_run.cost_aproximacio)
        print("CONVERGENCE: ", convergence)

    def __runDEModel(self, function):
        # funcio que retorna el resultat de simular amb el model de differential evolution

        # Si peta al iniciar i diu que no ha trobat solució vàlida, pot ser que no hi hagi un popsize prou gran
        """
        result = differential_evolution(func=function,
                                        popsize=150,  # 160
                                        bounds=self.varbound,
                                        integrality=[True] * len(self.varbound),
                                        maxiter=self.maxiter,
                                        mutation=(0.15, 0.25),  # 0.075, 0.15
                                        recombination=0.7,  # 0.7
                                        tol=0.0001,  # 0.02
                                        # to prevent from stopping when we can improve and there are iterations left
                                        strategy='best1bin',
                                        init="halton",
                                        disp=True,
                                        callback=self.__updateDEStep,
                                        workers=-1)  # -1

        print("Status: ", result['message'])
        print("Total evaluations: ", result['nfev'])
        print("Solution: ", result['x'])
        print("Cost: ", result['fun'])
        """
        return #result  # retornem la solucio (configuracio de vbounds)

    def __obtainHidrogenStationInfo(self):

        # Search hidrogen station
        n_consumers = 0
        hidrogen_total_storage = 0
        hidrogen_min_capacity = 0
        hidrogen_max_capacity = 0
        hidro_station_start = 0
        hidro_station_end = 0
        hidro_power_range = [0, 0]

        hidrogen_station: HidrogenStation

        for consumer in self.solucio_run.consumers.values():
            if isinstance(consumer, HidrogenStation):

                hidrogen_station = consumer
                hidrogen_total_storage = hidrogen_station.totalStorage()
                hidrogen_max_capacity = hidrogen_station.maxCapacity()
                hidrogen_min_capacity = hidrogen_station.minCapacity()

                hidro_station_start = consumer.vbound_start
                hidro_station_end = consumer.vbound_end
                hidro_power_range = consumer.calendar_range

                break

            n_consumers += 1

        return hidro_station_start, hidro_station_end, hidrogen_total_storage, hidrogen_min_capacity, hidrogen_max_capacity, hidro_power_range

    def costBH(self, configuracio):

        try:
            return self.costSA(configuracio)
        except KeyError:
            return 999999999999

    def costSA(self, configuracio):

        for variable in range(len(configuracio)):
            configuracio[variable] = round(configuracio[variable])

        return self.costDE(configuracio)

    def costPSO(self, configuracio):
        """
        Funcio de cost completa on s'optimitza totes les variables possibles. (Electrolizer, HVAC, ev_chargers )
        """

        for particle in range(len(configuracio)):
            for i in range(len(configuracio[particle])):
                configuracio[particle][i] = round(configuracio[particle][i])

            configuracio[particle].astype(int)

        best_particle = []
        for particle in configuracio:

            balanc_energetic_per_hores, cost, total_hidrogen_kg, numero_assets_ok, \
                consumers_individual_profile, generators_individual_profile, es_states = self.__calcBalanc(particle)

            self.kwargs_for_simulating.clear()

            if len(balanc_energetic_per_hores) == 0:
                self.__clearAssetsInfo()
                best_particle.append(cost)

            else:
                self.solucio_run.balanc_energetic_per_hores = balanc_energetic_per_hores
                self.solucio_run.numero_assets_ok = numero_assets_ok

                injectat = balanc_energetic_per_hores.copy()
                for n in range(0, len(injectat)):  # treiem positius
                    if injectat[n] > 0:
                        injectat[n] = 0
                    else:
                        injectat[n] = abs(injectat[n])  # valor absolut

                consumit = balanc_energetic_per_hores.copy()
                for n in range(0, len(consumit)):  #
