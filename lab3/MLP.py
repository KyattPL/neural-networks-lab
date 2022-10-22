import numpy as np
import csv


class MLP:

    def __init__(self, layers, neuronsInLayers, activationFuncs) -> None:
        self.howManyLayers = layers
        self.activationFuncs = activationFuncs
        self.weights = []
        self.biases = []
        self.stimulations = []
        self.activations = []

        for index in range(len(neuronsInLayers) - 1):
            cur = neuronsInLayers[index]
            next = neuronsInLayers[index + 1]
            stdDeviation = np.random.random()
            self.weights.append(np.random.normal(0, stdDeviation, (cur, next)))

            self.biases.append(np.random.normal(0, stdDeviation, next))

    def calc_outputs(self, input):
        stimulations = []
        activations = []
        stimulations.append(np.matmul(input, self.weights[0]) + self.biases[0])
        activations.append(self.calc_activations(
            stimulations[0], self.activationFuncs[0]))

        for i in range(1, self.howManyLayers - 1):
            stimulations.append(
                np.matmul(stimulations[i - 1], self.weights[i]) + self.biases[i])
            activations.append(self.calc_activations(
                stimulations[i], self.activationFuncs[i]))

        self.stimulations = stimulations
        self.activations = activations

    def calc_activations(self, stimulated, activationFunc):
        return activationFunc(stimulated)

    def save_to_csv(self):
        np.savez('weights.csv', *self.weights)

    def read_from_csv(self):
        files = np.load('weights.csv.npz')
        self.weights = []
        for name in files.files:
            self.weights.append(files[name])
