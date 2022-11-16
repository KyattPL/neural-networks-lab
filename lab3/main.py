from utils import *
from MLP import MLP
from keras.datasets import mnist
import matplotlib.pyplot as plt
import sys

NEURONS_INSIDE = 20
DATASET_SIZE = 60_000
TEST_SIZE = 10_000
BATCH_SIZE = 60
EPOCH_NUM = 10
EPSILON = 1000
LEARNING_COEF = 1e-5
IS_EARLY_STOPPING = True
REPEATS_NUM = 10
STANDARD_DEV = 0.001
FUNC = sigmoid if sys.argv[1] == "sigmoid" else hyper_tangent if sys.argv[1] == "htan" else relu if sys.argv[1] == "relu" else softplus

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
                    784, NEURONS_INSIDE, 10], activationFuncs=[FUNC, softmax],
                    standardDev=STANDARD_DEV, batchSize=BATCH_SIZE, learningCoef=LEARNING_COEF)


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
        howManyTimesBroken = 0

        for i in range(10):
            confusionMatrix.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        while np.power(current - prev, 2) > EPSILON:
            prev = current
            i = 0
            x_train, y_train = shuffle_training_data(x_train, y_train)
            predictions = []
            prevWeights = network.weights

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
                network.update_weights(batch_x)

            vectorized = [label_to_vector(l) for l in y_train]
            print(f'Cost function: {cost_whole(predictions, vectorized)}')
            
            current = cost_whole(predictions, vectorized)
            network.save_to_csv()

            if IS_EARLY_STOPPING and current > prev:
                network.weights = prevWeights
                howManyTimesBroken += 1
                if howManyTimesBroken == 3:
                    break

            epochs += 1
            epochsList.append(epochs)
            acc = accuracy(network, x_test, y_test)
            errorsList.append(1 - acc)

        accuracy(network, x_test, y_test, confusionMatrix)
        errors10Runs.append(errorsList)
        epochs10Runs.append(epochsList)
        confusions10Runs.append(confusionMatrix)

    
    confSummed = np.empty((10, 10))
    for matrix in confusions10Runs:
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

    avgEpochs = 0
    for ep in epochs10Runs:
        avgEpochs += len(ep)
    avgEpochs /= REPEATS_NUM

    errorsSummed = []
    epochsSummed = []
    for i in range(int(avgEpochs)):
        avgCurr = 0
        epochs = 0
        for run in errors10Runs:
            if i < len(run):
                avgCurr += run[i]
                epochs += 1
        errorsSummed.append(avgCurr)
        epochsSummed.append(epochs)

    errorsAvg = []
    for i in range(int(avgEpochs)):
        errorsAvg.append(errorsSummed[i] / epochsSummed[i])

    print(confSummed / REPEATS_NUM)
    plt.plot(np.array(epochs10Runs[indexHighest]), np.array(errors10Runs[indexHighest]), color='r', label='max')
    plt.plot(np.array(np.arange(1, int(avgEpochs) + 1, 1)), np.array(errorsAvg), color='b', label='avg')
    plt.plot(np.array(epochs10Runs[indexLowest]), np.array(errors10Runs[indexLowest]), color='g', label='min')

    plt.title(f"Odchylenie standardowe: {STANDARD_DEV}")
    plt.xlabel("Liczba epok")
    plt.ylabel("Błąd")

    plt.show()