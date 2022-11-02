import numpy as np
from utils import get_derivative, get_derivative_single,max_label

BATCH_SIZE = 500
LEARNING_COEF = 0.0001

class MLP:

    def __init__(self, layers, neuronsInLayers, activationFuncs, standardDev) -> None:
        self.howManyLayers = layers
        self.neuronsInLayers = neuronsInLayers
        self.activationFuncs = activationFuncs
        self.weights = []
        self.biases = []
        self.stimulations = []
        self.activations = []
        self.errors = []

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

    # correct -> [[1, 0, 0, 0, 0, 0, 0, 0], ...] -> czyli lista P list (tyle ile wzorców)
    #                                           gdzie każda lista ma K labelek (dla każdego neurona)
    def calc_errors(self, correct, inputIndex):
        end_layer_index = self.howManyLayers - 1
        act_fun_index = end_layer_index - 1
        hidden_layers = self.howManyLayers - 2

        derivative = get_derivative_single(self.activationFuncs[act_fun_index])
        errors_in_input = []
        errors_layer = []
        for k in range(self.neuronsInLayers[end_layer_index]):
            predicted = self.activations[inputIndex][end_layer_index - 1][k]
            label = correct[k]
            delta = (label - predicted) * derivative(self.stimulations[inputIndex][end_layer_index - 1][k])

            errors_layer.append(delta)
        
        errors_in_input.append(errors_layer)
        for l in range(hidden_layers):
            errors_layer = []
            derivative = get_derivative_single(self.activationFuncs[act_fun_index - 1 - l])
            weights = self.weights[end_layer_index - 1 - l]
            for k in range(self.neuronsInLayers[end_layer_index - 1 - l]):
                dx = derivative(self.stimulations[inputIndex][end_layer_index - 2 - l][k])
                deltas_weights = np.sum(np.dot(errors_in_input[l], weights[k]))
                delta = deltas_weights * dx
                errors_layer.append(delta)
            errors_in_input.append(errors_layer)

        errors_in_input.reverse()
        self.errors.append(errors_in_input)

    
    def update_weights(self, inputs):
        first_weights = self.weights[0]
        self.weights[0] += LEARNING_COEF * np.multiply(inputs, self.errors)
        # for r in range(len(first_weights)):
        #     for c in range(len(first_weights[r])):
        #         increase = 0
        #         for i in range(BATCH_SIZE):
        #             increase += inputs[i][r] * self.errors[i][0][c]
        #         self.weights[0][r][c] += LEARNING_COEF * increase

        for l in range(self.howManyLayers - 2):
            for r in range(self.weights[1 + l]):
                for c in range(self.weights[1 + l][r]):
                    increase = 0
                    for i in range(BATCH_SIZE):
                        increase += self.activations[i][1 + l][c] * self.errors[i][1 + l][c]
                    self.weights[1 + l][r][c] += LEARNING_COEF * increase

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
