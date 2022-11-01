import numpy as np


def linear(stimulations):
    return stimulations


def dx_linear(stimulations):
    return np.ones(len(stimulations))


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


def sigmoid_single(z):
    return 1 / (1 + np.power(np.e, -z))

def dx_sigmoid(stimulations):
    results = []
    for stim in stimulations:
        results.append(sigmoid_single(stim) * (1 - sigmoid_single(stim)))
    
    return results


def hyper_tangent(stimulations):
    activated = []
    for z in stimulations:
        activated.append(-1 + (2 / (1 + np.power(np.e, -2 * z))))

    return activated


def htan_single(z):
    return -1 + (2 / (1 + np.power(np.e, -2 * z)))

def dx_hyper_tangent(stimulations):
    results = []
    for stim in stimulations:
        results.append(1 - np.power(htan_single(stim), 2))

    return results


def relu(stimulations):
    activated = []

    for z in stimulations:
        if z < 0:
            activated.append(0)
        else:
            activated.append(z)

    return activated


def relu_single(z):
    return 0 if z < 0 else z


def dx_relu(stimulations):
    results = []
    for stim in stimulations:
        if stim > 0:
            results.append(1)
        else:
            results.append(0)

    return results


def softplus(stimulations):
    activations = []
    for z in stimulations:
        activations.append(np.log(1 + np.power(np.e, z)))

    return activations


def dx_softplus(stimulations):
    results = []
    for z in stimulations:
        results.append(1 / (1 + np.power(np.e, -z)))

    return results


def get_derivative(act_fn):
    if act_fn == linear:
        return dx_linear
    elif act_fn == sigmoid:
        return dx_sigmoid
    elif act_fn == hyper_tangent:
        return dx_hyper_tangent
    elif act_fn == relu:
        return dx_relu
    else:
        return dx_softplus


def max_label(outputs):
    maxIndex = 0
    maxPred = outputs[0]
    
    for (index, output) in enumerate(outputs):
        if output > maxPred:
            maxPred = output
            maxIndex = index
    
    return maxIndex


def NLL(predictions):
    results = []
    for pred in predictions:
        results.append(-np.log(pred))
    
    return results