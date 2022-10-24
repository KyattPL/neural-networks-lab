from utils import *
from MLP import MLP
from keras.datasets import mnist

#     O  O
#    /\ /\
#   O  O  O
#  / \ /\ /\
# O  O  O  O


if __name__ == "__main__":
    testInp = [0, 1, 0]

    network = MLP(layers=4, neuronsInLayers=[
                  3, 4, 4, 2], activationFuncs=[relu, relu, softmax],
                  standardDev=0.1)
    network.read_from_csv()

    print("Weights:")
    print(network.weights)

    network.calc_outputs(testInp)

    print("Stimulations:")
    print(network.stimulations)
    print("Activations:")
    print(network.activations)

    print("\nPredicted label:")
    print(f'\t{max_label(network.activations[-1])}')