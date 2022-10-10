import numpy as np
from random import randint, random

DATA_SIZE = 100
DIST_FROM_OG = 0.2


def rand_and_num():
    sign = randint(0, 1)
    sign_snd = randint(0, 1)

    if sign == 0:
        sign = -1

    if sign_snd == 0:
        sign_snd = -1

    return (randint(0, 1) + random() * DIST_FROM_OG * sign) * sign_snd


def rand_and_num_bipolar():
    sign = randint(0, 1)
    sign_snd = randint(0, 1)

    if sign == 0:
        sign = -1

    if sign_snd == 0:
        sign_snd = -1

    return (1 + random() * DIST_FROM_OG * sign) * sign_snd


def generate_data():
    inputs = np.zeros((DATA_SIZE, 2))
    outputs = np.zeros(DATA_SIZE)

    for i in range(DATA_SIZE):
        inputs[i] = [rand_and_num(), rand_and_num()]
        if inputs[i][0] > 0.8 and inputs[i][1] > 0.8:
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
        if inputs[i][0] > 0.8 and inputs[i][1] > 0.8:
            outputs[i] = 1
        else:
            outputs[i] = -1

    np.savetxt("input_bipolar.csv", inputs, delimiter=',')
    np.savetxt("output_bipolar.csv", outputs, delimiter=',')


if __name__ == "__main__":
    generate_data()
    generate_data_bipolar()
