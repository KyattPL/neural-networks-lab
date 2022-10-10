from random import randint, random
import numpy as np

DATA_SIZE = 100
TRAIN_PERCENT = 0.8
TRAIN_SIZE = int(DATA_SIZE * TRAIN_PERCENT)
TEST_SIZE = DATA_SIZE - TRAIN_SIZE

# DIST_FROM_OG = 0.1
WEIGHT_INIT = 0.001
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

    print(f"Correct: {correct}/{TEST_SIZE}")


def experiment():
    inputs = np.loadtxt("input.csv", delimiter=',')
    outputs = np.loadtxt("output.csv", delimiter=',')

    train, test = inputs[:TRAIN_SIZE], inputs[TRAIN_SIZE:]
    trainOut, testOut = outputs[:TRAIN_SIZE], outputs[TRAIN_SIZE:]

    weights = np.zeros(3)

    for i in range(3):
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
            stimulation += weights[2]
            predicted = activation_unipolar(stimulation)
            delta = trainOut[i] - predicted
            if delta != 0:
                isEpochRunning = True
            weights[0] += LEARNING_COEF * delta * train[i][0]
            weights[1] += LEARNING_COEF * delta * train[i][1]
            weights[2] += LEARNING_COEF * delta
        epochNum += 1

    print(f"Epoch: {epochNum}")
    check_test_data(test, weights, testOut)
    return weights


if __name__ == "__main__":
    trained_weights = experiment()
    print(trained_weights)
    while True:
        x1 = float(input("x1: "))
        x2 = float(input("x2: "))
        stimulation = trained_weights[0] * x1 + trained_weights[1] * x2
        stimulation += trained_weights[2]
        result = activation_unipolar(stimulation)
        print(f"Prediction: {result}\n")
