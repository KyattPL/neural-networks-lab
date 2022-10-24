import numpy as np


def linear(stimulations):
    return stimulations


def softmax(stimulations):
    denominator = 0

    for z in stimulations:
        denominator += np.power(np.e, z)

    softmaxed = []
    for z in stimulations:
        softmaxed.append(np.power(np.e, z) / denominator)

    return softmaxed


def sigmoid(stimulations):
    activated = []
    for z in stimulations:
        activated.append(1 / (1 + np.power(np.e, -z)))

    return activated


def hyper_tangent(stimulations):
    activated = []
    for z in stimulations:
        activated.append(-1 + (2 / (1 + np.power(np.e, -2 * z))))

    return activated


def relu(stimulations):
    activated = []

    for z in stimulations:
        if z < 0:
            activated.append(0)
        else:
            activated.append(z)

    return activated


def max_label(outputs):
    maxIndex = 0
    maxPred = outputs[0]
    
    for (index, output) in enumerate(outputs):
        if output > maxPred:
            maxPred = output
            maxIndex = index
    
    return maxIndex