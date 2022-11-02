from utils import *
from MLP import MLP
from keras.datasets import mnist

DATASET_SIZE = 60_000
BATCH_SIZE = 500

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = np.reshape(x_train, (60_000, 784))
    x_test = np.reshape(x_test, (10_000, 784))

    network = MLP(layers=4, neuronsInLayers=[
                  784, 500, 500, 10], activationFuncs=[relu, relu, softmax],
                  standardDev=0.01)

    i = 0
    while i < DATASET_SIZE:
        network.activations = []
        network.stimulations = []
        batch_x = x_train[i * BATCH_SIZE : (i+1) * BATCH_SIZE]
        batch_y = y_train[i * BATCH_SIZE : (i+1) * BATCH_SIZE]

        for j in range(BATCH_SIZE):
            network.calc_outputs(batch_x[j])
            network.calc_errors(label_to_vector(batch_y[j]), j)
        
        # Update weights

    # print("Weights:")
    # print(network.weights)

    # network.calc_outputs(testInp)

    # print("Stimulations:")
    # print(network.stimulations)
    # print("Activations:")
    # print(network.activations)

    # print("\nPredicted label:")
    # print(f'\t{max_label(network.activations[-1])}')