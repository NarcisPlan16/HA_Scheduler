# Interface for the Consumer entities. To abstract the simulate method so we can change it on run-time

from abc import abstractmethod


class AbsConsumer:
    """
    Class that is the parent for the different energy consumers
    """
    def __init__(self, configuration, name):
        # Initialize the consumer with the given configuration and name
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
        '''
        Method to simulate the consumer
        :param calendar: the calendar
        :param kwargs: dictionary with all the extra necessary arguments
        :return: None
        '''
        pass

    @abstractmethod
    def resetToInit(self):
        '''
        Method to reset the consumer to its initial state
        :return: None
        '''
        pass

    @abstractmethod
    def hasTanks(self):
        '''
        Method to check if the consumer has tanks
        :return: True if the consumer has tanks, False otherwise
        '''
        pass
