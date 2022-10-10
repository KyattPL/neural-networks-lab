from random import randint, random
import numpy as np

DATA_SIZE = 100
TRAIN_PERCENT = 0.8
TRAIN_SIZE = int(DATA_SIZE * TRAIN_PERCENT)
TEST_SIZE = DATA_SIZE - TRAIN_SIZE

# DIST_FROM_OG = 0.2
WEIGHT_INIT = 0.001
LEARNING_COEF = 0.01
#THETA = 0.1


def activation_unipolar(stimulation, theta):
    return 1 if stimulation > theta else 0


def check_test_data(testData, weights, correctLabels, theta):
    correct = 0
    for i in range(TEST_SIZE):
        stimulation = weights[0] * testData[i][0] + weights[1] * testData[i][1]
        predicted = activation_unipolar(stimulation, theta)
        delta = correctLabels[i] - predicted
        if delta == 0:
            correct += 1

    #print(f"Correct: {correct}/{TEST_SIZE}")


def experiment(theta):
    inputs = np.loadtxt("input.csv", delimiter=',')
    outputs = np.loadtxt("output.csv", delimiter=',')

    train, test = inputs[:TRAIN_SIZE], inputs[TRAIN_SIZE:]
    trainOut, testOut = outputs[:TRAIN_SIZE], outputs[TRAIN_SIZE:]

    weights = np.zeros(2)

    for i in range(2):
        sign = randint(0, 1)
        if sign == 0:
            sign = -1
        weights[i] = sign * random() * WEIGHT_INIT

    epochNum = 1
    isEpochRunning = True
    while isEpochRunning:

        isEpochRunning = False
        for i in range(TRAIN_SIZE):
            stimulation = weights[0] * train[i][0] + weights[1] * train[i][1]
            predicted = activation_unipolar(stimulation, theta)
            delta = trainOut[i] - predicted
            if delta != 0:
                isEpochRunning = True

            weights[0] += LEARNING_COEF * delta * train[i][0]
            weights[1] += LEARNING_COEF * delta * train[i][1]
        epochNum += 1

    check_test_data(test, weights, testOut, theta)
    return epochNum, weights


if __name__ == "__main__":
    thetas = [0.05, 0.1, 0.15, 0.2, 0.25,
              0.3, 0.35, 0.4, 0.45, 0.5, 0.8, 2.5, 100.0]
    for theta in thetas:
        epochSum = 0
        weightsSum = np.zeros(2)
        print(f"Theta: {theta}")

        for i in range(10):
            epochs, weights = experiment(theta)
            epochSum += epochs
            weightsSum += weights

        print(f"\tAvg epochs num: {epochSum / 10}")
        print(f"\tAvg weights: {weightsSum / 10}")
