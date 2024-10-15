# Class for the wind turbines

from Generator import Generator


class WT(Generator):

    def __init__(self, configuration, name):
        super().__init__(configuration, name)
