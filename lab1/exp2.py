from random import randint, random
import numpy as np

DATA_SIZE = 100
TRAIN_PERCENT = 0.8
TRAIN_SIZE = int(DATA_SIZE * TRAIN_PERCENT)
TEST_SIZE = DATA_SIZE - TRAIN_SIZE

# DIST_FROM_OG = 0.1
#WEIGHT_INIT = 0.1
LEARNING_COEF = 0.01


def activation_unipolar(stimulation):
    return 1 if stimulation > 0 else 0


def check_test_data(testData, weights, correctLabels):
    correct = 0
    for i in range(TEST_SIZE):
        stimulation = weights[0] * testData[i][0] + weights[1] * testData[i][1]
        stimulation += weights[2]
        predicted = activation_unipolar(stimulation)
        delta = correctLabels[i] - predicted
        if delta == 0:
            correct += 1

    #print(f"Correct: {correct}/{TEST_SIZE}")


def experiment(weight_init):
    inputs = np.loadtxt("input.csv", delimiter=',')
    outputs = np.loadtxt("output.csv", delimiter=',')

    train, test = inputs[:TRAIN_SIZE], inputs[TRAIN_SIZE:]
    trainOut, testOut = outputs[:TRAIN_SIZE], outputs[TRAIN_SIZE:]

    weights = np.zeros(3)

    for i in range(3):
        sign = randint(0, 1)
        if sign == 0:
            sign = -1
        weights[i] = sign * random() * weight_init

    epochNum = 1
    isEpochRunning = True
    while isEpochRunning:
        isEpochRunning = False
        for i in range(TRAIN_SIZE):
            stimulation = weights[0] * train[i][0] + weights[1] * train[i][1]
            stimulation += weights[2]
            predicted = activation_unipolar(stimulation)
            delta = trainOut[i] - predicted
            if delta != 0:
                isEpochRunning = True
            weights[0] += LEARNING_COEF * delta * train[i][0]
            weights[1] += LEARNING_COEF * delta * train[i][1]
            weights[2] += LEARNING_COEF * delta
        epochNum += 1

    #print(f"Epoch: {epochNum}")
    check_test_data(test, weights, testOut)
    return epochNum, weights


if __name__ == "__main__":
    weightInits = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    for weightInit in weightInits:
        epochSum = 0
        weightSum = np.zeros(3)
        print(f"Weights: {weightInit * -1} to {weightInit}")

        for i in range(10):
            epochs, weights = experiment(weightInit)
            epochSum += epochs
            weightSum += weights

        print(f"\tAvg epochs num: {epochSum / 10}")
        print(f"\tAvg weights: {weightSum / 10}")
