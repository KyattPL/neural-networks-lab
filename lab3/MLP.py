import numpy as np


class MLP:

    def __init__(self, layers, neuronsInLayers, activationFuncs, standardDev) -> None:
        self.howManyLayers = layers
        self.activationFuncs = activationFuncs
        self.weights = []
        self.biases = []
        self.stimulations = []
        self.activations = []

        for index in range(len(neuronsInLayers) - 1):
            cur = neuronsInLayers[index]
            next = neuronsInLayers[index + 1]
            self.weights.append(np.random.normal(0, standardDev, (cur, next)))
            self.biases.append(np.random.normal(0, standardDev, next))

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
        np.savez('biases.csv', *self.biases)

    def read_from_csv(self):
        weightsData = np.load('weights.csv.npz')
        self.weights = []
        for name in weightsData.files:
            self.weights.append(weightsData[name])
        
        biasesData = np.load('biases.csv.npz')
        self.biases = []
        for name in biasesData.files:
            self.biases.append(biasesData[name])
