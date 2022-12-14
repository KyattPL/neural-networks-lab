from random import randint, random
import numpy as np

DATA_SIZE = 100
TRAIN_PERCENT = 0.8
TRAIN_SIZE = int(DATA_SIZE * TRAIN_PERCENT)
TEST_SIZE = DATA_SIZE - TRAIN_SIZE


WEIGHT_INIT = 0.001
ALLOWED_ERR = 0.4


def activation_bipolar(stimulation):
    return 1 if stimulation > 0 else -1


def check_test_data(testData, weights, correctLabels):
    correct = 0
    for i in range(TEST_SIZE):
        stimulation = weights[0] * testData[i][0] + \
            weights[1] * testData[i][1] + weights[2]
        predicted = activation_bipolar(stimulation)
        delta = correctLabels[i] - predicted
        if delta == 0:
            correct += 1

    #print(f"Correct: {correct}/{TEST_SIZE}")


def experiment(learningCoef):
    inputs = np.loadtxt("input_bipolar.csv", delimiter=',')
    outputs = np.loadtxt("output_bipolar.csv", delimiter=',')

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
        sum_errors = 0

        for i in range(TRAIN_SIZE):
            stimulation = weights[0] * train[i][0] + weights[1] * train[i][1]
            stimulation += weights[2]
            delta = trainOut[i] - stimulation
            sum_errors += delta * delta
            weights[0] += learningCoef * delta * train[i][0]
            weights[1] += learningCoef * delta * train[i][1]
            weights[2] += learningCoef * delta

        if sum_errors / TRAIN_SIZE < ALLOWED_ERR:
            isEpochRunning = False

        epochNum += 1

    check_test_data(test, weights, testOut)
    return epochNum, weights


if __name__ == "__main__":
    learningCoefs = [0.3, 0.2, 0.1, 0.05, 0.025, 0.01, 0.001, 0.0001, 0.00001]
    for learningCoef in learningCoefs:
        epochSum = 0
        weightsSum = np.zeros(3)
        print(f"Learning coef: {learningCoef}")

        for i in range(10):
            epochs, weights = experiment(learningCoef)
            epochSum += epochs
            weightsSum += weights

        print(f"\tAvg epochs num: {epochSum / 10}")
        print(f"\tAvg weights: {weightsSum / 10}")