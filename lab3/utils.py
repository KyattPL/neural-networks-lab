import numpy as np


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
