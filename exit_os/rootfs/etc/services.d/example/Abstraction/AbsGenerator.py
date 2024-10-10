# Class that is the parent for the different energy generators

from abc import abstractmethod

class AbsGenerator:
    """
    Class that is the parent for the different energy generators
    """

    def __init__(self, configuration, name):
        # Initialize the generator with the given configuration and name
        self.maxWattage = configuration['max_output']
        self.minWattage = configuration['min_output']
        self.name = name
        self.config = configuration
        self.calendar_range = configuration['calendar_range']
        self.active_hours = configuration['active_hours']
        self.active_calendar = configuration['active_calendar']
        self.production = configuration['production']
        self.controllable = configuration['controllable']
        self.vbound_start = 0
        self.vbound_end = 0
        # Here the common attributes


    @abstractmethod
    def doSimula(self, calendar, **kwargs):
    '''
    Method to simulate the generator
    :param calendar: the calendar
    :param kwargs: dictionary with all the extra necessary arguments
    :return: None
    '''
        pass

    def obtainProductionByHour(self, hour):
    '''
    Method to obtain the production of the generator for a given hour
    :param hour: the hour of the day
    :return: the production of the generator for that hour
    '''
        return self.production[hour]

    def obtainProduction(self):
    '''
    Method to obtain the production of the generator
    :return: the production of the generator
    '''
        return self.production

    def obtainDailyProduction(self):
    '''
    Method to obtain the daily production of the generator
    :return: the daily production of the generator
    '''
        return sum(self.production)
