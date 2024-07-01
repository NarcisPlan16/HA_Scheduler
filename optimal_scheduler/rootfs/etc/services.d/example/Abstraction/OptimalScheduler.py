# Optimal scheduler "class". Main program
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

ha_url = "http://192.168.0.117:8123"
bearer_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJmZThlNTgyNDBhYTA0M2UwOTYyMmRmZWJlMTc5MDc0YyIsImlhdCI6MTcxOTMwNDY4NiwiZXhwIjoyMDM0NjY0Njg2fQ.j8euYQxDWMkJJqHNpTXUBE1rrhpOm1Vr-WcY3fdt8q0"
# If for some reason, we are getting a code 401 (Unauthorized) and previously we didn0t, try creating a new bearer token

headers = {
    "Authorization": f"Bearer {bearer_token}",
    "Content-Type": "application/json",
}

logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Example usage
#logger.info("AAAAAAAAAAAAAAAAAA")

class OptimalScheduler:
    """
    Optimal scheduler class. This class is the responsable for the optimization of the assets. Uses teh forecasters to perform the optimization.

    Attributes
    -----------
    Here are only the most relevant attributes.
    maxiter : int
        Maximum nimber of iterations for the model to perform the optimization.
    hores_simular : int
        Number of hours to perform the optimization.
    electricity_prices: list
        Hourly buying prices of the electricity (Automaticly calculated using an API).
    electricity_sell_prices: list
        Hourly selling prices of the electricity.
    latitude: float
        Latitude where we want to get the meto data.
    longitude: float
        Longitude where we want to get the meto data.
    meteo_data: DataFrame
        Pandas dataframe with the meteorological data prediction for the next day and the data of today. 
        For example, if predicting the next -hores_simular- hours, this parameter must have size (-hores_simular- * 2, n).
    kwargs_for_simulating: dict
        Dictionary with arguments to pass to the simulation code. This parameters helps the simulation code of each asset get the desired information 
        and enables comunication between assets as you can tore here values and retrieve them with another asset simulation code.
    """

    def __init__(self):

        self.connections = []
        self.progress = []  # Array with the best cost value on each step
        self.debug = True  # Indicates if we are on debug mode
        self.console_debug = False  # Indicates if we want to run and debug the cody by console.
        # If set to true and not executing from elan_os folder, we will get errors
        self.maxiter = 30  # 20
        # Parameter for the configurator to get the correct path depending on where we execute the code

        self.hores_simular = 24
        self.hidrogen_price = 1.6
        self.consumer_invalid_solutions = 0  # Total invalid solution for the consumers, just for debug purposes
        self.generator_invalid_solutions = 0

        self.electricity_prices = self.__obtainElectricityPrices()
        self.electricity_sell_prices = [0.054, 0.054, 0.054, 0.054, 0.054, 0.054, 0.054, 0.054, 0.054, 0.054, 0.054, 0.054,
                                        0.054, 0.054, 0.054, 0.054, 0.054, 0.054, 0.054, 0.054, 0.054, 0.054, 0.054, 0.054]
        
        self.latitude = "41.963138"
        self.longitude = "2.831640"
        self.meteo_data = ForecastersManager.obtainMeteoData(self.latitude, self.longitude)
        self.electrcity_production_forecast = []
        self.electricity_consumption_forecast = []

        self.solucio_run = Solution()
        self.solucio_final = Solution()

        self.assets = {'Buildings': {'Consumption': [], 'Generation': []}, 'Consumers': {}, 'EnergySources': {}, 'Generators': {}}
        # TODO: Millorar perquè crei una entrada per cada folder a Asset_types automaticament

        self.kwargs_for_simulating = {}
        # key arguments for those assets that share a common restriction and
        # one execution effects the others assets execution


    def __obtainElectricityPrices(self):
        
        tomorrow = datetime.today() + timedelta(1)
        tomorrow_str = tomorrow.strftime('%Y%m%d')
        url = f"https://www.omie.es/es/file-download?parents%5B0%5D=marginalpdbc&filename=marginalpdbc_{tomorrow_str}.1"


        response = requests.get(url)
        if response.status_code != "200":
            today = datetime.today().strftime('%Y%m%d')
            url = f"https://www.omie.es/es/file-download?parents%5B0%5D=marginalpdbc&filename=marginalpdbc_{today}.1"
            response = requests.get(url)

        with open("omie_price_pred.csv", 'wb') as f:
            f.write(response.content)

        hourly_prices = []
        with open('omie_price_pred.csv', 'r') as file:
            for line in file.readlines()[1:-1]:
                components = line.strip().split(';')
                components.pop(-1) # delete blank character at the end
                hourly_price = float(components[-1])
                hourly_prices.append(hourly_price)

        return hourly_prices
    
    def __preparePredData(self, type: str):
        """
        Prepares the data for the prediction type specified as parameter -type-

        Parameters
        -----------
        type : str
            String specifying the prediction type.
            * Consumption
            * Generation
        -----------
        Returns
        -----------
        Returns a DataFrame with the data needed for the specified type of prediction. For now, only a Dataframe with Timestamp - State 
        but on new versions would be good to add all relevant attributes so the forecasters can perform better.
        """
        
        dictionary = {'Timestamp': [], 'state': []}
        res = pd.DataFrame(dictionary)

        today = datetime.today().replace(hour=0, minute=0, second=0)
        tomorrow = datetime.today() + timedelta(days=1)
        start_of_tomorrow = datetime(tomorrow.year, tomorrow.month, tomorrow.day)

        today_str =  (today + pd.Timedelta(hours=1)).strftime('%Y-%m-%d')
        tomorrow_str = tomorrow.strftime('%Y-%m-%d')

        for building_type in self.solucio_run.buildings[type]:  # each building type (Consumption or Generation)

            response = requests.get(
                    f"{ha_url}/api/history/period/" + today_str + "T00:00:00?end_time=" + tomorrow_str + "T00:00:00&filter_entity_id=" + building_type,
                    headers=headers)
                            
            response_data = response.json()[0]
            data = pd.DataFrame()
            data = data.from_dict(response_data)

            state_data = data['state']
            if len(data['state']) < 24:  # No available data for past 24h
                
                print("[WARNING]: No data found for previous 24h. Searching previous data and using its mean value")
                #TODO mean = self.__calcMissingValueFiller(today, building_type) 
                mean = 0

                for i in range (0, 24):
                    date = pd.to_datetime(today + timedelta(hours=i)).strftime('%Y-%m-%d %H:%M:%S')
                    res.loc[len(res.index)] = [date, mean]

            else: # Available data for past 24h
                for i in range (0, 24): # Add previos hours data rows
                    date = pd.to_datetime(today + timedelta(hours=i)).strftime('%Y-%m-%d %H:%M:%S')
                    res.loc[len(res.index)] = [date, state_data[i]]

            # Add prediction rows
            for i in range (0, 24):
                date = pd.to_datetime(start_of_tomorrow + timedelta(hours=i)).strftime('%Y-%m-%d %H:%M:%S')
                res.loc[len(res.index)] = [date, 0]
        
        return res
    
    def __calcMissingValueFiller(self, today, building_type):

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
                for n in range(0, len(consumit)):  # treiem negatius
                    if consumit[n] < 0:
                        consumit[n] = 0

                # €€€€
                preus_consumit = np.multiply(consumit, np.array(self.electricity_prices))
                preus_injectat = np.multiply(injectat, np.array(self.electricity_sell_prices))
                cost_per_hores = np.subtract(preus_consumit, preus_injectat)

                self.solucio_run.cost_per_hours = [0] * self.hores_simular
                self.solucio_run.cost_per_hours = cost_per_hores

                preu = sum(cost_per_hores) - total_hidrogen_kg * self.hidrogen_price

                self.solucio_run.saveConsumersProfileData(consumers_individual_profile)
                self.solucio_run.saveGeneratorsProfileData(generators_individual_profile)
                self.solucio_run.saveEnergySourcesStates(es_states)
                self.solucio_run.preu_cost = preu
                self.solucio_run.cost_aproximacio = cost

                self.__clearAssetsInfo()

                best_particle.append(preu + cost)

        return best_particle

    def costDE(self, configuracio):
        """
        Funcio de cost completa on s'optimitza totes les variables possibles. (Electrolizer, HVAC, ev_chargers )
        """

        balanc_energetic_per_hores, cost, total_hidrogen_kg, numero_assets_ok, \
            consumers_individual_profile, generators_individual_profile, es_states = self.__calcBalanc(configuracio)

        self.kwargs_for_simulating.clear()

        if len(balanc_energetic_per_hores) == 0:
            self.__clearAssetsInfo()
            return cost

        self.solucio_run.balanc_energetic_per_hores = balanc_energetic_per_hores
        self.solucio_run.numero_assets_ok = numero_assets_ok

        injectat = balanc_energetic_per_hores.copy()
        for n in range(0, len(injectat)):  # treiem positius
            if injectat[n] > 0:
                injectat[n] = 0
            else:
                injectat[n] = abs(injectat[n])  # valor absolut

        consumit = balanc_energetic_per_hores.copy()
        for n in range(0, len(consumit)):  # treiem negatius
            if consumit[n] < 0:
                consumit[n] = 0

        # €€€€
        preus_consumit = np.multiply(consumit, np.array(self.electricity_prices))
        preus_injectat = np.multiply(injectat, np.array(self.electricity_sell_prices))
        cost_per_hores = np.subtract(preus_consumit, preus_injectat)

        self.solucio_run.cost_per_hours = [0] * self.hores_simular
        self.solucio_run.cost_per_hours = cost_per_hores

        preu = sum(cost_per_hores) - total_hidrogen_kg * self.hidrogen_price

        self.solucio_run.saveConsumersProfileData(consumers_individual_profile)
        self.solucio_run.saveGeneratorsProfileData(generators_individual_profile)
        self.solucio_run.saveEnergySourcesStates(es_states)

        self.solucio_run.preu_cost = preu
        self.solucio_run.cost_aproximacio = cost

        self.__clearAssetsInfo()

        return preu + cost

    def __calcConsumersBalance(self, config):

        self.kwargs_for_simulating.clear()

        consumers_total_profile = [0] * self.hores_simular  # perfil total de consum hora a hora
        consumers_individual_profile = {}  # diccionari amb key = nom del consumer i valor = consumption profile
        consumers_total_kwh = 0  # total de kwh gastats
        total_hidrogen_kg = 0  # total d'hidrogen generat en kg

        cost_aproximacio = 0
        numero_de_consumer = 0  # Variable per dir que com a minim alguna configuracio dels assets era bona
        # com més gran, més bona la solució

        consumer: AbsConsumer
        for consumer_class in self.solucio_run.consumers:  # for every consumer class (EVCharger, HidrogenStation...)
            for consumer in self.solucio_run.consumers[consumer_class].values():  # for every consumer

                start = consumer.active_calendar[0]
                end = consumer.active_calendar[1] + 1
                self.kwargs_for_simulating['electricity_prices'] = self.electricity_prices[start:end]

                res_dictionary = consumer.doSimula(calendar=config[consumer.vbound_start: consumer.vbound_end],
                                                   kwargs_simulation=self.kwargs_for_simulating)

                consumption_profile, consumed_Kwh, total_hidrogen_kg, cost = self.__unpackSimulationResults(res_dictionary)
                # TODO: Maillorar. Ara mateix l'unpack és un grapat d'ifs quan en realitat
                #  ens retornen un diccionari de resultats

                # simulam l'asset. total_hidrogen_kg == 0 si no és una hidrogen station
                consumers_total_kwh += consumed_Kwh
                cost_aproximacio += cost

                if len(consumption_profile) < self.hores_simular:  # solucio no vàlida

                    consumers_total_profile.pop()  # per dir que la solucio no és vàlida
                    self.kwargs_for_simulating.clear()

                    return consumers_total_profile, consumers_individual_profile, consumers_total_kwh, \
                        numero_de_consumer, cost_aproximacio, total_hidrogen_kg

                numero_de_consumer += 1  # Una configuració bona més
                consumers_individual_profile[consumer.name] = consumption_profile

                for i in range(0, len(consumers_total_profile)):  # per cada hora
                    consumers_total_profile[i] += consumption_profile[i]

        self.kwargs_for_simulating.clear()

        return consumers_total_profile, consumers_individual_profile, consumers_total_kwh, numero_de_consumer, cost_aproximacio, total_hidrogen_kg

    def __calcGeneratorsBalance(self, config):
        # funcio per calcular el nou balanç energetic tenint en compte el que generen els generadors

        self.kwargs_for_simulating.clear()

        generator: AbsGenerator
        generators_total_kwh = 0
        generators_total_profile = [0] * self.hores_simular
        generators_individual_profile = {}
        cost_aproximacio = 0
        numero_de_generator = 0

        for generator_class in self.solucio_run.generators:  # per cada classe de generador
            for generator in self.solucio_run.generators[generator_class].values():  # per cada generador

                if not generator.controllable:
                    generators_total_kwh += generator.obtainDailyProduction()
                    for i in range(0, len(generators_total_profile)):  # per cada hora
                        generators_total_profile[i] += generator.obtainProductionByHour(i)

                    generators_individual_profile[generator.name] = generator.obtainProduction()

                else:

                    res_dictionary = generator.doSimula(calendar=config[generator.vbound_start: generator.vbound_end],
                                                        kwargs_simulation=self.kwargs_for_simulating)
                    
                    production_profile, produced_Kwh, _, cost = self.__unpackSimulationResults(res_dictionary)
                    generators_total_kwh += produced_Kwh
                    cost_aproximacio += cost

                    if len(production_profile) < self.hores_simular:  # solucio no vàlida

                        generators_total_profile.pop()  # per dir que la solucio no és vàlida
                        self.kwargs_for_simulating.clear()

                        return generators_total_profile, generators_individual_profile, generators_total_kwh,  numero_de_generator, \
                                cost_aproximacio

                    generators_individual_profile[generator.name] = production_profile

                    for i in range(0, len(generators_total_profile)):  # per cada hora
                        generators_total_profile[i] += production_profile[i]

                numero_de_generator += 1

        self.kwargs_for_simulating.clear()

        return generators_total_profile, generators_individual_profile, generators_total_kwh, numero_de_generator, cost_aproximacio

    def __calcEnergySourcesBalance(self, config, balanc_energetic_per_hores):

        # TODO: solucio ja te balanc_energetic_per_hores però com que ara tenim problemes de consistènca
        #  amb la solucio, el posarem de parametre temporalment

        self.kwargs_for_simulating.clear()

        energy_source: AbsEnergySource
        bat_states = {}
        numero_energy_source = 0
        energy_sources_total_profile = balanc_energetic_per_hores
        cost = 0

        for es_class in self.solucio_run.energy_sources:  # per cada classe d'energy source
            for energy_source in self.solucio_run.energy_sources[es_class].values():  # per cada energy source

                start = energy_source.vbound_start
                end = energy_source.vbound_end

                ini = energy_source.active_calendar[0]
                fi = energy_source.active_calendar[1] + 1
                self.kwargs_for_simulating['electricity_prices'] = self.electricity_prices[ini:fi]

                res_dictionary = energy_source.doSimula(calendar=config[start: end],
                                                        kwargs_simulation=self.kwargs_for_simulating)

                consumption_profile, perfil_carrega, _, cost = self.__unpackSimulationResults(res_dictionary)
                # TODO: Maillorar. Ara mateix l'unpack és un grapat d'ifs quan en realitat
                #  ens retornen un diccionari de resultats
                # consumption profile: Valors negatius = descarreguem bateria.
                #                      Valors positius = carreguem (xarxa + excedent de generacio)

                if len(consumption_profile) < 24:
                    self.kwargs_for_simulating.clear()
                    return energy_sources_total_profile[0:len(energy_sources_total_profile) - 1], numero_energy_source, cost, []

                for hour in range(0, len(energy_sources_total_profile)):
                    energy_sources_total_profile[hour] += consumption_profile[hour]

                bat_states[energy_source.name] = perfil_carrega
                numero_energy_source += 1

        self.kwargs_for_simulating.clear()

        return energy_sources_total_profile, numero_energy_source, cost, bat_states

    def __calcBalanc(self, configuracio):

        # cost de tots els consumers
        consumers_total_profile, consumers_individual_profile, consumers_total_kwh, valid_ones, \
            cost_aproximacio, total_hidrogen_kg = self.__calcConsumersBalance(configuracio)

        if len(consumers_total_profile) < self.hores_simular:  # solucio no valida
            # si hem simulat menys de les hores demanades, vol dir que la solució no és vàlida
            self.consumer_invalid_solutions += 1
            return [], (9999999999999 - (valid_ones * 100) + cost_aproximacio), total_hidrogen_kg, valid_ones + 1, {}, {}, []

        # ---------------------------------------------------------- #
        #
        #       De moment tots els GENERATORS es comptaràn com 
        #       si fossin el building production. En un futur
        #       s'hauria de fer que es fes el forecast per cada 
        #       un i llavors tenir el seu perfil individual de 
        #       producció.
        #
        # ---------------------------------------------------------- #

        # kwh produits pels generators
        generators_total_profile, generators_individual_profile, generators_total_kwh, gen_valids, \
            cost_aproximacio_gen = self.__calcGeneratorsBalance(configuracio)
        
        if len(generators_total_profile) < self.hores_simular:  # solucio no valida
            self.generator_invalid_solutions += 1
            return [], (9999999999999 - (valid_ones * 10) - (gen_valids * 10) + cost_aproximacio + cost_aproximacio_gen), \
                total_hidrogen_kg, (valid_ones + gen_valids + 1), {}, {}, []

        # Add the building consuming costs
        consumers_total_kwh += self.building_consumption['state'].sum()
        hour = 0
        for index in self.building_consumption.index.tolist():
            consumers_total_profile[hour] += self.building_consumption['state'][index]
            hour += 1

        # ---------------------------------------------------------- #
        #
        #       De moment tots els GENERATORS es comptaràn com 
        #       si fossin el building production. En un futur
        #       s'hauria de fer que es fes el forecast per cada 
        #       un i llavors tenir el seu perfil individual de 
        #       producció.
        #
        # ---------------------------------------------------------- #

        # Add the building production 
        generators_total_kwh += self.building_production['state'].sum()
        hour = 0
        for index in self.building_production.index.tolist():
            generators_total_profile[hour] += self.building_production['state'][index]
            hour += 1

        balanc_energetic_per_hores = np.subtract(consumers_total_profile, generators_total_profile)
        # negatius és injectar + consumir

        balanc_energetic_per_hores, bat_valids, \
            cost_aproximacio_bat, es_states = self.__calcEnergySourcesBalance(configuracio, balanc_energetic_per_hores)

        if len(balanc_energetic_per_hores) < 24:
            return [], (9999999999999 - (valid_ones * 10) - (bat_valids * 10) - (gen_valids * 10) + cost_aproximacio + \
                cost_aproximacio_gen + cost_aproximacio_bat), total_hidrogen_kg, (valid_ones + gen_valids + bat_valids + 1), {}, {}, []

        return balanc_energetic_per_hores, (cost_aproximacio + cost_aproximacio_gen+ cost_aproximacio_bat), total_hidrogen_kg, \
            (valid_ones + gen_valids + bat_valids), consumers_individual_profile, generators_individual_profile, es_states

    def mostrarResultat(self, temps):

        start = 0
        end = 0

        print("\n--------------Variables del model--------------")
        print("Consumers:\n")
        for consumer_class in self.solucio_final.consumers:
            for consumer in self.solucio_final.consumers[consumer_class].values():

                print(consumer.name)

                end += consumer.active_calendar[1] + 1
                print(self.solucio_final.model_variables[start:end])
                start = end + 1

                print("\n")

        for source_class in self.solucio_final.energy_sources:
            for source in self.solucio_final.energy_sources[source_class].values():

                print(source.name)

                end += source.active_calendar[1] + 1
                print(self.solucio_final.model_variables[start:end])
                start = end + 1

                print("\n")

        print("--------------Resultats de la predicció--------------")
        for consumer in self.solucio_final.consumption_data:
            print(consumer)
            print(self.solucio_final.consumption_data[consumer])
            print("\n")

        for source_class in self.solucio_final.energy_sources:
            for source in self.solucio_final.energy_sources[source_class]:
                print(source)
                print(self.solucio_final.energy_sources_data[source])
                print("\n")

        print("---Preu per hores sense comptar l'hidrogen---")
        print(self.solucio_final.cost_per_hours)
        print("Sumat: ", sum(self.solucio_final.cost_per_hours))
        print("\n")

        print("---Consum per hores total---")
        print(self.solucio_final.balanc_energetic_per_hores) 
        # TODO: Fix. Now it does not work. The array is empty. Possible fix to try: do no save the "solution" and only save the model.
        #       When the algorithm finishes, take the model and calculate the data for every asset using the variable states.
        print("\n")

        print("****************************************************")
        print("INVALID SOLUTIONS FOUND: ", self.consumer_invalid_solutions)
        print("COST FINAL: ", self.solucio_final.preu_cost, "€")
        print("COST APROXIMACIO: ", self.solucio_final.cost_aproximacio)
        print("TEMPS TARDAT: ", temps, "s")
        print("CAPACITATS DELS TANKS: ", self.solucio_final.tanks_final_capacity)
        print("****************************************************")
        process = psutil.Process()
        memoria = process.memory_info().rss / 1000000  # in MBytes
        print("Memory used for this process: ", memoria, "MB")
        print("****************************************************")

    def __configureBounds(self):

        # numero de variables * numero d'hores a simular(per cada asset)
        varbound = []
        index = 0

        consumer: AbsConsumer
        for asset_class in self.solucio_run.consumers:  # for every consumer class (EVCharger, HidrogenStation...)
            for consumer in self.solucio_run.consumers[asset_class].values():  # for every consumer

                consumer.vbound_start = index

                for hour in range(consumer.active_calendar[0], consumer.active_calendar[1] + 1):
                    varbound.append([consumer.calendar_range[0], consumer.calendar_range[1]])
                    index += 1

                consumer.vbound_end = index

        generator: AbsGenerator
        for asset_class in self.solucio_run.generators:  # for every generator class (PV, WT...)
            for generator in self.solucio_run.generators[asset_class].values():  # for every consumer

                if generator.controllable:
                    generator.vbound_start = index

                    for hour in range(generator.active_calendar[0], generator.active_calendar[1] + 1):
                        varbound.append([generator.calendar_range[0], generator.calendar_range[1]])
                        index += 1

                    generator.vbound_end = index

        energy_source: AbsEnergySource
        for asset_class in self.solucio_run.energy_sources:  # for every energy source class (battery...)
            for energy_source in self.solucio_run.energy_sources[asset_class].values():  # for every energy source

                energy_source.vbound_start = index

                max_discharge = energy_source.calendar_range[0]
                max_charge = energy_source.calendar_range[1]

                for hour in range(energy_source.active_calendar[0], energy_source.active_calendar[1] + 1):
                    varbound.append([max_discharge, max_charge])
                    index += 1

                energy_source.vbound_end = index

        return np.array(varbound)

    def startOptimization(self, write_pipe):

        self.connections.append(write_pipe)

        self.solucio_run = Solution(self.assets['Buildings'], self.assets['Consumers'], self.assets['EnergySources'], self.assets['Generators'])
        self.solucio_final = Solution(self.assets['Buildings'], self.assets['Consumers'], self.assets['EnergySources'], self.assets['Generators'])
        self.varbound = self.__configureBounds()

        # Uncomment for debug
        #if self.console_debug:
        #    prof = cProfile.Profile()
        #    prof.enable()

        temps_inici = time.time()
        result = self.__optimize()
        temps_fi = time.time()

        #if self.console_debug:
        #    prof.disable()
        #    prof.print_stats(sort='cumtime')

        self.solucio_final.model_variables = result.x
        self.solucio_final.temps_tardat = temps_fi - temps_inici

        for connection in self.connections:
            connection.send([-1, self.maxiter])
            connection.send(self.solucio_final)
            connection.close()

        self.connections.clear()

        x_values = range(1, len(self.progress)+1)
        print(x_values)

        #plt.plot(x_values, self.progress)
        #plt.grid()
        #plt.xlabel("Iteration")
        #plt.ylabel("Cost (€)")
        #plt.title("Cost over iterations")

        #fig1 = plt.gcf()

        #current_dir = os.getcwd()
        #if self.console_debug:
        #    current_dir = os.path.join(current_dir, "Abstraction")
        #img_dir = os.path.join(current_dir, "result_imgs", "cost.png")
        #os.makedirs("result_imgs", exist_ok=True)
        #fig1.savefig(img_dir, dpi=200)

        #plt.show()

        self.mostrarResultat(temps_fi - temps_inici)

    def startOptimizationNoPipe(self):

        self.solucio_run = Solution(self.assets['Buildings'], self.assets['Consumers'], self.assets['EnergySources'], self.assets['Generators'])
        self.solucio_final = Solution(self.assets['Buildings'], self.assets['Consumers'], self.assets['EnergySources'], self.assets['Generators'])
        self.varbound = self.__configureBounds()

        # Uncomment for debug
        #if self.console_debug:
        #    prof = cProfile.Profile()
        #    prof.enable()

        temps_inici = time.time()
        result = self.__optimize()
        temps_fi = time.time()

        #if self.console_debug:
        #    prof.disable()
        #    prof.print_stats(sort='cumtime')

        self.__postResultsToHA(result)

        print("Result: ")
        print(result)
        self.solucio_final.model_variables = result #result.x for the DE algorithm
        self.solucio_final.temps_tardat = temps_fi - temps_inici

        x_values = range(1, len(self.progress)+1)
        print(x_values)

        self.mostrarResultat(temps_fi - temps_inici)

    def __postResultsToHA(self, results):
        # TODO: Add API POST to schedule the results for each asset optimization

        pass

    def __unpackSimulationResults(self, res_dictionary: dict):

        cons_profile = []
        consumption = 0
        hidrogen_kg = 0
        cost = 0

        if res_dictionary.keys().__contains__('consumption_profile'):
            cons_profile = res_dictionary['consumption_profile']

        if res_dictionary.keys().__contains__('consumed_Kwh'):
            consumption = res_dictionary['consumed_Kwh']

        if res_dictionary.keys().__contains__('total_generated_kg'):
            hidrogen_kg = res_dictionary['total_generated_kg']

        if res_dictionary.keys().__contains__('cost_aproximacio'):
            cost = res_dictionary['cost_aproximacio']

        return cons_profile, consumption, hidrogen_kg, cost

    def __clearAssetsInfo(self):

        consumer: AbsConsumer
        for consumer_class in self.solucio_run.consumers:
            for consumer in self.solucio_run.consumers[consumer_class].values():
                consumer.resetToInit()

        source: AbsEnergySource
        for source_class in self.solucio_run.energy_sources:
            for source in self.solucio_run.energy_sources[source_class].values():
                source.resetToInit()

    def addAsset(self, asset_type, asset_class, asset_config):
        # asset type = "generator"... asset_class = name of the asset/class file name

        configurator = Configurator(console=self.console_debug)
        asset = configurator.configureAndCreate(asset_class, asset_config)

        if not self.assets[asset_type].__contains__(asset_class):
            self.assets[asset_type][asset_class] = {}

        self.assets[asset_type][asset_class][asset.name] = asset

    def AddBuilding(self, type, id):

        if not self.assets["Buildings"][type].__contains__(id):
            self.assets["Buildings"][type].append(id)

    def obtainAssetsInfo(self):

        dict_res = {}
        for asset_type in self.assets.keys():  # for every asset type (Consumers, Generators...)

            dict_res[asset_type] = {}
            for asset_class in self.assets[asset_type].keys():  # for every asset class (EVCharger, PV...)

                dict_res[asset_type][asset_class] = {}
                for asset in self.assets[asset_type][asset_class].keys():  # for every asset
                    dict_res[asset_type][asset_class][asset] = self.assets[asset_type][asset_class][asset].config

        return dict_res

    def savePrices(self, sell, buy):

        self.electricity_sell_prices = sell
        self.electricity_prices = buy

    def saveAssetsConfigurationInfo(self, name):

        res = {}
        for asset_type in self.assets:  # for every asset type (Consumers, Generators...)

            res[asset_type] = {}
            for asset_class in self.assets[asset_type]:  # for every class (EVCharger, HidrogenStation, PV...)

                res[asset_type][asset_class] = {}
                for asset in self.assets[asset_type][asset_class].values():  # for every asset of this type

                    res[asset_type][asset_class][asset.name] = asset.config  # save the config

        file = open(name, "w")
        json.dump(res, file, indent=4)  # write the content to the file
        file.close()

    def deleteAssets(self):

        for asset_type in self.assets:
            for asset_class in self.assets[asset_type]:
                self.assets[asset_type][asset_class].clear()
                # Clear all the assets of the class of that type
                # (Ex: clear all EVChargers from the Consumers dictionary)
