# Photovoltaic class for photovoltaic generators

from Generator import Generator


class PV(Generator):

    def __init__(self, configuration, name):
        super().__init__(configuration, name)
