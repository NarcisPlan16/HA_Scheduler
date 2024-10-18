# OptimalAssets.py

class AbsConsumer:
    def __init__(self, configuration, name):
        self.config = configuration
        self.name = name
        #self.max_power = configuration['max_power']
        self.calendar_range = configuration['calendar_range']
        self.active_hours = configuration['active_hours']
        self.active_calendar = configuration['active_calendar']

        self.vbound_start = 0
        self.vbound_end = 0
    
    @abstractmethod
    def doSimula(self, calendar, **kwargs):
        pass

    @abstractmethod
    def resetToInit(self):
        pass

    @abstractmethod
    def hasTanks(self):
        pass

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


class AbsEnergySource:
    def __init__(self, configuration, name):
        self.cap_actual = configuration['cap_actual']
        self.max_capacity = configuration['max_capacity']
        self.max_roc = configuration['max_roc']
        self.max_rod = configuration['max_rod']
        self.calendar_range = configuration['calendar_range']
        self.active_hours = configuration['active_hours']
        self.active_calendar = configuration['active_calendar']
        self.name = name
        self.config = configuration
        self.vbound_start = 0
        self.vbound_end = 0
        # here the common attributes between energy sources

    @abstractmethod
    def doSimula(self, calendar, **kwargs):
        pass

    @abstractmethod
    def canviaSimula(self, simImpl):
        pass

    @abstractmethod
    def resetToInit(self):
        pass