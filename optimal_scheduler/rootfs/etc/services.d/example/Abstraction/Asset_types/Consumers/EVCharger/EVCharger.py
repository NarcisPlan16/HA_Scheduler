# EVCharger class
import os

from AbsConsumer import AbsConsumer
from utils.utils import createClass


class EVCharger(AbsConsumer):

    def __init__(self, configuration, name):

        super().__init__(configuration, name)
        self.simul = createClass("SIMU_" + name, configuration['Simulate'])

    def doSimula(self, **kwargs):
        self.config['calendar'] = kwargs['calendar']
        return self.simul.simula(self.simul, self.config, kwargs_simulation=kwargs['kwargs_simulation'])

    def canviaSimula(self, simImpl):  # simImpl will be an implementation of the Simulate interface
        self.simul = simImpl
