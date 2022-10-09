import numpy as np
from random import randint, random

DATA_SIZE = 40
DIST_FROM_OG = 0.1


def rand_and_num():
    sign = randint(0, 1)
    if sign == 0:
        sign = -1
    return (randint(0, 1) + random() * DIST_FROM_OG) * sign


def rand_and_num_bipolar():
    sign = randint(0, 1)
    if sign == 0:
        sign = -1
    return (1 + random() * DIST_FROM_OG) * sign


def generate_data():
    inputs = np.zeros((DATA_SIZE, 2))
    outputs = np.zeros(DATA_SIZE)

    for i in range(DATA_SIZE):
        inputs[i] = [rand_and_num(), rand_and_num()]
        # XD? Skąd mam wiedzieć jak blisko tego prawego górnego mam uznawać za 1?
        if inputs[i][0] > 0.9 and inputs[i][1] > 0.9:
            outputs[i] = 1
        else:
            outputs[i] = 0

    np.savetxt("input.csv", inputs, delimiter=',')
    np.savetxt("output.csv", outputs, delimiter=',')


def generate_data_bipolar():
    inputs = np.zeros((DATA_SIZE, 2))
    outputs = np.zeros(DATA_SIZE)

    for i in range(DATA_SIZE):
        inputs[i] = [rand_and_num_bipolar(), rand_and_num_bipolar()]
        # XD? Skąd mam wiedzieć jak blisko tego prawego górnego mam uznawać za 1?
        if inputs[i][0] > 0.9 and inputs[i][1] > 0.9:
            outputs[i] = 1
        else:
            outputs[i] = -1

    np.savetxt("input_bipolar.csv", inputs, delimiter=',')
    np.savetxt("output_bipolar.csv", outputs, delimiter=',')


if __name__ == "__main__":
    generate_data()
    generate_data_bipolar()
