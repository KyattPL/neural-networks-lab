from utils import *
from MLP import MLP
from keras.datasets import mnist
import matplotlib.pyplot as plt

DATASET_SIZE = 60_000
TEST_SIZE = 10_000
BATCH_SIZE = 500
EPOCH_NUM = 10
EPSILON = 1000
LEARNING_COEF = 1e-5
IS_EARLY_STOPPING = True
REPEATS_NUM = 3

def shuffle_training_data(x_train, y_train):
    perm = np.random.permutation(len(x_train))
    return x_train[perm], y_train[perm]


if __name__ == "__main__":
    print("1. Continue with existing weights")
    print("2. New training")
    print("3. New with OG weights")
    choice = int(input().strip())

    errors10Runs = []
    epochs10Runs = []
    confusions10Runs = []

    for i in range(REPEATS_NUM):

        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = np.reshape(x_train, (DATASET_SIZE, 784))
        x_test = np.reshape(x_test, (TEST_SIZE, 784))

        network = MLP(layers=3, neuronsInLayers=[
                    784, 10, 10], activationFuncs=[sigmoid, softmax],
                    standardDev=0.001, batchSize=BATCH_SIZE, learningCoef=LEARNING_COEF)


        if choice == 1:
            network.read_from_csv()
        elif choice == 3:
            network.read_from_csv('_og')

        current = 0
        prev = float("inf")
        prevWeights = None
        epochs = 0
        epochsList = []
        errorsList = []
        confusionMatrix = []
        for i in range(10):
            confusionMatrix.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        while np.power(current - prev, 2) > EPSILON:
            prev = current
            i = 0
            x_train, y_train = shuffle_training_data(x_train, y_train)
            predictions = []

            while i < DATASET_SIZE / BATCH_SIZE:
                network.activations = []
                network.stimulations = []
                network.errors = []
                batch_x = x_train[i * BATCH_SIZE : (i+1) * BATCH_SIZE]
                batch_y = y_train[i * BATCH_SIZE : (i+1) * BATCH_SIZE]

                for j in range(BATCH_SIZE):
                    network.calc_outputs(batch_x[j])
                    network.calc_errors(label_to_vector(batch_y[j]), j)
                    predictions.append(network.activations[-1][-1])

                i += 1
                prevWeights = network.weights
                network.update_weights(batch_x)

            vectorized = [label_to_vector(l) for l in y_train]
            print(f'Cost function: {cost_whole(predictions, vectorized)}')
            
            current = cost_whole(predictions, vectorized)
            network.save_to_csv()

            if IS_EARLY_STOPPING and current > prev:
                network.weights = prevWeights

            epochs += 1
            epochsList.append(epochs)
            acc = accuracy(network, x_test, y_test, confusionMatrix)
            errorsList.append(1 - acc)

        errors10Runs.append(errorsList)
        epochs10Runs.append(epochsList)
        confusions10Runs.append(confusionMatrix)

    
    confSummed = None
    for matrix in confusionMatrix:
        confSummed += np.array(matrix)

    indexHighest = 0
    indexLowest = 0
    maxErr = errors10Runs[0][-1]
    minErr = errors10Runs[0][-1]
    
    for (i, errors) in enumerate(errors10Runs):
        if errors[-1] > maxErr:
            maxErr = errors[-1]
            indexHighest = i
        elif errors[-1] < minErr:
            minErr = errors[-1]
            indexLowest = i

    print(confSummed / REPEATS_NUM)
    plt.plot(np.array(epochs10Runs[indexHighest]), np.array(errors10Runs[indexHighest]), color='r', label='max')
    plt.plot(np.array(epochsList[0]), np.array(errorsList[0]), color='b', label='avg')
    plt.plot(np.array(epochs10Runs[indexLowest]), np.array(errors10Runs[indexLowest]), color='g', label='min')

    plt.title("Błąd na zbiorze testowym")
    plt.xlabel("Liczba epok")
    plt.ylabel("Błąd")

    plt.show()