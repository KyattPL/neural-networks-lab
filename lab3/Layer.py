import numpy as np


class Layer:

    def __init__(self, size, activationFunc=None) -> None:
        self.neurons = np.empty(size)
        self.activationFunc = activationFunc
