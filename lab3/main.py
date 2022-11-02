from utils import *
from MLP import MLP
from keras.datasets import mnist

DATASET_SIZE = 60_000
TEST_SIZE = 10_000
BATCH_SIZE = 500

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = np.reshape(x_train, (DATASET_SIZE, 784))
    x_test = np.reshape(x_test, (TEST_SIZE, 784))

    network = MLP(layers=4, neuronsInLayers=[
                  784, 400, 200, 10], activationFuncs=[relu, relu, softmax],
                  standardDev=0.1)

    i = 0
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
        
        i += 1
        network.update_weights(batch_x)
    
    correct = 0
    for i in range(TEST_SIZE):
        activs = network.test_input(x_test[i])
        label = max_label(activs)
        if label == y_test[i]:
            correct += 1
    
    print(f'Correct {correct} / 10000')
    print(f'Percentage: {(correct / TEST_SIZE) * 100}%')