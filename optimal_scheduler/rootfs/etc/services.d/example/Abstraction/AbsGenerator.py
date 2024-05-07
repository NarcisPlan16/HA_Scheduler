# Class that is the parent for the different energy generators

from abc import abstractmethod

class AbsGenerator:

    def __init__(self, configuration, name):
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
        pass

    def obtainProductionByHour(self, hour):
        return self.production[hour]

    def obtainProduction(self):
        return self.production

    def obtainDailyProduction(self):
        return sum(self.production)
