# Class to represent the configuration actions

import os
import sys
import tomli

# Set up directory paths for module imports
current_dir = os.path.dirname(os.path.abspath("Configurator.py"))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Append subfolder directories to the system path for importing
subfolder_dir = os.path.join(parent_dir, 'utils')
sys.path.append(subfolder_dir)

subfolder_dir = os.path.join(parent_dir, 'Asset_types')
sys.path.append(subfolder_dir)

from utils import utils


class Configurator:
    def __init__(self, console):
        # Initialize the console and assets directory
        self.console = console
        self.assets_dir = os.getcwd()  # Get the current working directory
        
        # If console is True, modify the assets directory path
        if self.console:
            self.assets_dir = os.path.join(self.assets_dir, "Abstraction")

    def __enterAssets(self, dir_list):
        # Remove any .toml files from the list
        for element in dir_list:
            if element.__contains__(".toml"):
                dir_list.remove(element)
                break

        assets = {}  # Dictionary to store asset names and their counts
        total_elements = 0
        
        # Loop through directories and collect asset names and counts
        for asset in dir_list:
            if not os.path.isdir(asset):  # List only directories (assets)
                asset_name = asset.replace(".py", "")
                print("     -> " + asset)
                print("How many assets of this type are there?")
                
                # Input asset count and validate it
                assets[asset_name] = input("Enter the number: ")
                while not assets[asset_name].isdigit():
                    assets[asset_name] = input()
                assets[asset_name] = int(assets[asset_name])  # Convert string to int

                total_elements += assets[asset_name]
                print("\n")

        return assets, total_elements  # Return dictionary and total count

    def __readConfigData(self, path, asset_type, filename):
        # Read configuration data from a .toml file
        dir_path = os.path.join(path, asset_type, "")
        config_path = dir_path + filename + ".toml"  # Path to the .toml configuration file
        print(config_path)
        
        with open(config_path, "rb") as toml_file:
            config_data = tomli.load(toml_file)  # Read the file
            toml_file.close()  # Close the file
            print(config_data)
            print("\n")

        return config_data  # Return the read data

    def __configureAsset(self, names, path, asset_type, results):
        # Configure assets of a specific type based on their names
        for i in range(len(names)):
            filename = input(f"Enter the configuration filename in .toml format "
                             f"(ex: config1) for asset type {asset_type} - {names[i]}: ")
            done = False
            config_data = {}
            
            # Loop to read configuration data and handle errors
            while not done:
                try:
                    config_data = self.__readConfigData(path, asset_type, filename)  # Read the .toml configuration
                    done = True
                except tomli.TOMLDecodeError:
                    filename = input(f"Error loading configuration file {filename}. "
                                     f"Please re-enter its name: ")
                except FileNotFoundError:
                    filename = input(f"The configuration file {filename} does not exist. "
                                     f"Please re-enter its name: ")

            asset_path = os.path.join(path, asset_type)
            module_path = os.path.relpath(asset_path)  # Get relative path for the asset type
            class_instance = utils.createClass(module_path, asset_type)  # Obtain the class from the asset type string
            results[names[i]] = class_instance(config_data, names[i])  # Instantiate the class with config data and name

        return results

    def __configure(self, names, path):
        # Configure all assets for each asset type in the dictionary
        results = {}
        
        for asset_type in names:  # For each asset type in the dictionary
            print(f"----------------Entering configurations for assets of type {asset_type}----------------\n")
            results = self.__configureAsset(names[asset_type], path, asset_type, results)  # Configure assets and add to results

        return results

    def __enterNames(self, dictionary):
        # Prompt user for asset names based on counts in the provided dictionary
        names = {}
        for asset in dictionary:
            if dictionary[asset] > 0:
                names[asset] = []
                for n in range(dictionary[asset]):
                    names[asset].append(input(f"Enter name for asset number {n + 1} of type {asset}: "))

        return names

    def initializeModels(self):
        # SETUP
        print("Welcome to the SETUP, choose how many assets of each type your system has. "
              "(Buildings, Consumers, Generators, and EnergySources).\n")

        self.assets_dir = os.path.join(self.assets_dir, "Asset_types")
        current_dir = os.getcwd()

        # List available buildings
        print("--------------------Available Buildings-------------------- \n")
        buildings_list = os.listdir(os.path.join(self.assets_dir, "Buildings"))
        count_buildings, n_buildings = self.__enterAssets([buildings_list[0]])
        names_buildings = self.__enterNames(count_buildings)

        print("Buildings entered:")
        print(names_buildings)

        # List available consumers
        print("--------------------Available Consumers-------------------- \n")
        consumers_list = os.listdir(os.path.join(self.assets_dir, "Consumers"))
        count_consumers, n_consumers = self.__enterAssets(consumers_list)
        names_consumers = self.__enterNames(count_consumers)

        print("Consumers entered:")
        print(names_consumers)

        # List available generators
        print("--------------------Available Generators-------------------- \n")
        generators_list = os.listdir(os.path.join(self.assets_dir, "Generators"))
        count_generators, n_generators = self.__enterAssets(generators_list)
        names_generators = self.__enterNames(count_generators)

        print("Generators entered:")
        print(names_generators)

        # List available energy sources
        print("--------------------Available Energy Sources-------------------- \n")
        energy_sources_list = os.listdir(os.path.join(self.assets_dir, "EnergySources"))
        count_energy_sources, n_energy_sources = self.__enterAssets(energy_sources_list)
        names_sources = self.__enterNames(count_energy_sources)

        print("Energy Sources entered:")
        print(names_sources)

        print("\nNext, you will need to write the name of the configuration file for each asset\n")

        buildings = {}
        consumers = {}
        generators = {}
        energy_sources = {}
        
        # Configure buildings if any exist
        if n_buildings > 0:
            buildings = self.__configure(names_buildings, os.path.join(self.assets_dir, "Buildings"))

        # Configure consumers if any exist
        if n_consumers > 0:
            consumers = self.__configure(names_consumers, os.path.join(self.assets_dir, "Consumers"))

        # Configure generators if any exist
        if n_generators > 0:
            generators = self.__configure(names_generators, os.path.join(self.assets_dir, "Generators"))

        # Configure energy sources if any exist
        if n_energy_sources > 0:
            energy_sources = self.__configure(names_sources, os.path.join(self.assets_dir, "EnergySources"))

        return buildings, consumers, energy_sources, generators

    def configureAndCreate(self, asset_class, asset_config):
        # Create an asset instance based on configuration data
        class_instance = utils.createClass(asset_config["Class"])  # Get class from asset_config
        return class_instance(asset_config, asset_class)  # Instantiate the class and return it
