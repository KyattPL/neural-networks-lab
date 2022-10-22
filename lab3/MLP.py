import numpy as np
from Layer import Layer


class MLP:

    def __init__(self, layers, neuronsInLayers, activationFuncs, initWeight) -> None:
        self.howManyLayers = layers
        self.activationFuncs = activationFuncs
        # self.layers = np.empty(layers, dtype=Layer)
        self.weights = []
        self.stimulations = []
        self.activations = []

        # for (index, size) in enumerate(neuronsInLayers):
        #     if index != 0:
        #         self.layers[index] = Layer(size, activationFuncs[index - 1])
        #     else:
        #         self.layers[index] = Layer(size)

        for index in range(len(neuronsInLayers) - 1):
            cur = neuronsInLayers[index]
            next = neuronsInLayers[index + 1]
            self.weights.append(np.full((cur, next), initWeight))

    def calc_outputs(self, input):
        stimulations = []
        activations = []
        stimulations.append(np.matmul(input, self.weights[0]))
        activations.append(self.calc_activations(
            stimulations[0], self.activationFuncs[0]))

        for i in range(1, self.howManyLayers - 1):
            stimulations.append(
                np.matmul(stimulations[i - 1], self.weights[i]))
            activations.append(stimulations[i], self.activationFuncs[i])

        self.stimulations = stimulations
        self.activations = activations

    def calc_activations(self, stimulated, activationFunc):
        activations = []

        for num in stimulated:
            activations.append(activationFunc(num))

        return activations
