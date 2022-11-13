import numpy as np
from utils import get_derivative_single


class MLP:

    def __init__(self, layers, neuronsInLayers, activationFuncs, standardDev, batchSize, learningCoef) -> None:
        self.howManyLayers = layers
        self.neuronsInLayers = neuronsInLayers
        self.activationFuncs = activationFuncs
        self.weights = []
        self.biases = []
        self.stimulations = []
        self.activations = []
        self.errors = []
        self.batchSize = batchSize
        self.learningCoef = learningCoef

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

        self.stimulations.append(stimulations)
        self.activations.append(activations)

    def calc_errors(self, correct, inputIndex):
        end_layer_index = self.howManyLayers - 1
        act_fun_index = end_layer_index - 1
        hidden_layers = self.howManyLayers - 2

        derivative = get_derivative_single(self.activationFuncs[act_fun_index])
        errors_in_input = []

        if derivative != "softmax":
            errors_layer = []
            for k in range(self.neuronsInLayers[end_layer_index]):
                predicted = self.activations[inputIndex][end_layer_index - 1][k]
                label = correct[k]
                delta = (label - predicted) * derivative(self.stimulations[inputIndex][end_layer_index - 1][k])

                errors_layer.append(delta)
            errors_in_input.append(np.array(errors_layer))
        else:
            predicted = self.activations[inputIndex][end_layer_index - 1]
            dx = (np.array(correct) - np.array(predicted))
            p = dx * predicted
            errors_in_input.append(p)
        
        for l in range(hidden_layers):
            errors_layer = []
            derivative = get_derivative_single(self.activationFuncs[act_fun_index - 1 - l])
            weights = self.weights[end_layer_index - 1 - l]
            for k in range(self.neuronsInLayers[end_layer_index - 1 - l]):
                dx = derivative(self.stimulations[inputIndex][end_layer_index - 2 - l][k])
                deltas_weights = np.sum(np.dot(errors_in_input[l], weights[k]))
                delta = deltas_weights * dx
                errors_layer.append(delta)
            errors_in_input.append(np.array(errors_layer))

        errors_in_input.reverse()
        self.errors.append(np.array(errors_in_input, dtype=object))

    
    def update_weights(self, inputs):
        first_layer = []
        for arr in self.errors:
            first_layer.append(arr[0])

        self.weights[0] += self.learningCoef * np.matmul(np.array(inputs).transpose(), np.array(first_layer, dtype=np.float32))
        self.biases[0] += self.learningCoef * np.sum(np.array(first_layer, dtype=np.float32), axis=0)

        for l in range(self.howManyLayers - 2):
            nth_layer = []
            nth_activs = []
            for arr in self.errors:
                nth_layer.append(arr[l + 1])
            for arr in self.activations:
                nth_activs.append(arr[l])
            self.weights[1 + l] += self.learningCoef * np.matmul(np.array(nth_activs).transpose(), np.array(nth_layer, dtype=np.float32))
            self.biases[1 + l] += self.learningCoef * np.sum(np.array(nth_layer, dtype=np.float32), axis=0)


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


    def test_input(self, input):
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

        return activations[-1]