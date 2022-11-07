from utils import *
from MLP import MLP
from keras.datasets import mnist

DATASET_SIZE = 60_000
TEST_SIZE = 10_000
BATCH_SIZE = 500
EPOCH_NUM = 10
EPSILON = 1e-7

def shuffle_training_data(x_train, y_train):
    perm = np.random.permutation(len(x_train))
    return x_train[perm], y_train[perm]


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = np.reshape(x_train, (DATASET_SIZE, 784))
    x_test = np.reshape(x_test, (TEST_SIZE, 784))

    network = MLP(layers=3, neuronsInLayers=[
                  784, 10, 10], activationFuncs=[sigmoid, softmax],
                  standardDev=0.001, batchSize=BATCH_SIZE)

    print("1. Continue with existing weights")
    print("2. New training")
    choice = int(input().strip())

    if choice == 1:
        network.read_from_csv()

    current = 0
    prev = float("inf")
    while True:
    # while np.power(current - prev, 2) > EPSILON:
        prev = current
        i = 0
        x_train, y_train = shuffle_training_data(x_train, y_train)
        predictions = []

        while i < DATASET_SIZE / BATCH_SIZE:
            network.activations = []
            network.stimulations = []
            # TODO: clearować tu errory czy potrzebne do jakiegoś global checka nauki?
            network.errors = []
            batch_x = x_train[i * BATCH_SIZE : (i+1) * BATCH_SIZE]
            batch_y = y_train[i * BATCH_SIZE : (i+1) * BATCH_SIZE]

            for j in range(BATCH_SIZE):
                network.calc_outputs(batch_x[j])
                network.calc_errors(label_to_vector(batch_y[j]), j)
                predictions.append(network.activations[-1][-1])

            i += 1
            network.update_weights(batch_x)

        vectorized = [label_to_vector(l) for l in y_train]
        print(f'Cost function: {cost_whole(predictions, vectorized)}')
        
        network.save_to_csv()
        # current = min_cost_func(x_test, y_test, network)

        correct = 0
        for i in range(TEST_SIZE):
            activs = network.test_input(x_test[i])
            label = max_label(activs)
            if label == y_test[i]:
                correct += 1
        
        print(f'Correct {correct} / 10000')
        print(f'Percentage: {(correct / TEST_SIZE) * 100}%')


    correct = 0
    for i in range(TEST_SIZE):
        activs = network.test_input(x_test[i])
        label = max_label(activs)
        if label == y_test[i]:
            correct += 1
    
    print(f'Correct {correct} / 10000')
    print(f'Percentage: {(correct / TEST_SIZE) * 100}%')