from random import randint, random
import numpy as np

DATA_SIZE = 40
TRAIN_PERCENT = 0.8
TRAIN_SIZE = int(DATA_SIZE * TRAIN_PERCENT)
TEST_SIZE = DATA_SIZE - TRAIN_SIZE

DIST_FROM_OG = 0.1
#WEIGHT_INIT = 0.1
LEARNING_COEF = 0.2


# def rand_and_num():
#     sign = randint(0, 1)
#     if sign == 0:
#         sign = -1
#     return (randint(0, 1) + random() * DIST_FROM_OG) * sign


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
    # inputs = np.zeros((DATA_SIZE, 2))
    # outputs = np.zeros(DATA_SIZE)

    # for i in range(DATA_SIZE):
    #     inputs[i] = [rand_and_num(), rand_and_num()]
    #     # XD? Skąd mam wiedzieć jak blisko tego prawego górnego mam uznawać za 1?
    #     if inputs[i][0] > 0.9 and inputs[i][1] > 0.9:
    #         outputs[i] = 1
    #     else:
    #         outputs[i] = 0

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
    return epochNum


if __name__ == "__main__":
    weightInits = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    for weightInit in weightInits:
        epochSum = 0
        print(f"Weights: {weightInit * -1} to {weightInit}")

        for i in range(10):
            epochSum += experiment(weightInit)

        print(f"Avg epochs num: {epochSum / 10}")
