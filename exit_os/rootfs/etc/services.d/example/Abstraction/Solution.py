# Class to represent a possible solution configuration
from AbsConsumer import AbsConsumer
from AbsEnergySource import AbsEnergySource


class Solution:
    """
    Class to represent a possible solution configuration
    """

    def __init__(self, buildings={}, consumers={}, energy_sources={}, generators={}):
        # Initialize the solution with the given buildings, consumers, energy sources and generators
        self.generators = generators
        self.consumers = consumers
        self.energy_sources = energy_sources
        self.buildings = buildings

         # Initialize various attributes to track energy balance, costs, and capacity
        self.balanc_energetic_per_hores = []  # Stores energy balance for each hour
        self.numero_assets_ok = 0  # Counter for successful assets
        self.preu_cost = 9999999999  # Placeholder for cost price
        self.cost_aproximacio = 0  # Approximated cost
        self.temps_tardat = 0  # Delay time

        # Initialize data structures for consumption and production tracking
        self.consumption_data = {}
        for building_type in buildings.values():  # Iterate through building types
            for edifici in building_type:  # Iterate through each building
                self.consumption_data[edifici] = []  # Initialize consumption data for each building

        self.production_data = {}  # Data structure to store production data
        self.energy_sources_data = {}  # Data structure to store energy sources states
        self.tanks_final_capacity = {}  # Final capacity of tanks
        self.model_variables = []  # Variables used in the model

        self.cost_per_hours = []  # Stores cost data per hour

    def saveConsumersProfileData(self, profile):
        """
        Method to save the consumers profile data
        :param profile: dictionary with the profile data
        :return: None
        """
        consumer: AbsConsumer
        profile_aux = profile
        for consumer_class in self.consumers:  # we look every class of consumer
            for consumer in self.consumers[consumer_class].keys():

                if profile_aux.__contains__(consumer):

                    self.consumption_data[consumer] = profile_aux[consumer]

                    if self.consumers[consumer_class][consumer].hasTanks():
                        self.__saveTanksInfo(self.consumers[consumer_class][consumer])

                    profile_aux.pop(consumer)  # for efficiency purposes

    def saveGeneratorsProfileData(self, profile):
        """
        Method to save the generators profile data
        :param profile: dictionary with the profile data
        :return: None
        """
        for generator in profile:
            self.production_data[generator] = profile[generator]

    def saveEnergySourcesStates(self, profile):
        """
        Method to save the energy sources states
        :param profile: dictionary with the energy sources states
        :return: None
        """
        for key in profile:
            self.energy_sources_data[key] = profile[key]

    def __saveTanksInfo(self, consumer):
        """
        Method to save the tanks info
        :param consumer: consumer object
        :return: None
        """
        tanks_info = consumer.getTanksInfo()
        for tank in tanks_info:
            self.tanks_final_capacity[tank] = tanks_info[tank]

    def obtainAssetsProfiles(self):
        """
        Method to obtain the assets profiles
        :return: dictionary with the assets profiles
        """
        res_dict = {}  # Dictionary to store results

        # Iterate through consumers to gather their profiles
        for asset_class in self.consumers:
            res_dict[asset_class] = {}  # Initialize dictionary for the asset class
            for asset in self.consumers[asset_class]:
                res_dict[asset_class][asset] = {}  # Initialize dictionary for the asset

                asset_data = res_dict[asset_class][asset]  # Reference to the asset data
                consumer: AbsConsumer  # Type hint for consumer
                consumer = self.consumers[asset_class][asset]  # Get the consumer object

                # Calculate total consumption and prepare graph data
                asset_data['consumption'] = sum(self.consumption_data[asset])  # Total consumption
                asset_data["graph_data"] = self.consumption_data[asset]  # Data for graph
                asset_data["graph_y"] = "Consumption (kwh)"  # Y-axis label for graph
                asset_data["graph_x"] = "Hour"  # X-axis label for graph
                asset_data["graph_title"] = "Consumption by hour"  # Title for graph
                asset_data["control_variables"] = consumer.config["calendar"]  # Control variables

        # Iterate through energy sources to gather their profiles
        for asset_class in self.energy_sources:
            res_dict[asset_class] = {}  # Initialize dictionary for the asset class
            for asset in self.energy_sources[asset_class]:
                res_dict[asset_class][asset] = {}  # Initialize dictionary for the asset

                asset_data = res_dict[asset_class][asset]  # Reference to the asset data
                source: AbsEnergySource  # Type hint for energy source
                source = self.energy_sources[asset_class][asset]  # Get the energy source object

                consumption = []  # List to store consumption data
                asset_profile = self.energy_sources_data[asset]  # Get the profile data for the energy source
                for hour in range(len(asset_profile) - 1):  # Iterate through hours
                    consumption.append(asset_profile[hour] - asset_profile[hour + 1])  # Calculate consumption

                asset_data['consumption'] = sum(consumption)  # Total consumption for the asset
                asset_data["graph_data"] = asset_profile  # Data for graph
                asset_data["graph_y"] = "Battery capacity"  # Y-axis label for graph
                asset_data["graph_x"] = "Hour"  # X-axis label for graph
                asset_data["graph_title"] = "Battery capacity over time"  # Title for graph
                asset_data["control_variables"] = source.config["calendar"]  # Control variables

        return res_dict  # Return the collected asset profiles
