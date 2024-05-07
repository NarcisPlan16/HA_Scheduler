# Photovoltaic class for photovoltaic generators

from AbsGenerator import AbsGenerator
from utils.utils import createClass

class PV(AbsGenerator):

    def __init__(self, configuration, name):
        super().__init__(configuration, name)
        self.simul = createClass("SIMU_" + name, configuration['Simulate'])

    def doSimula(self, calendar, **kwargs):
        self.config['calendar'] = kwargs['calendar']
        return self.simul.simula(self.simul, self.config, kwargs_simulation=kwargs['kwargs_simulation'])
