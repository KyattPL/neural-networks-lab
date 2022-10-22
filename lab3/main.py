from utils import *
from MLP import MLP

#     O  O
#    /\ /\
#   O  O  O
#  / \ /\ /\
# O  O  O  O


if __name__ == "__main__":
    testInp = [0, 1, 1]

    network = MLP(layers=4, neuronsInLayers=[
                  3, 4, 4, 2], activationFuncs=[relu, relu, softmax])
    network.read_from_csv()
    print(network.weights)

    network.calc_outputs(testInp)

    print(network.activations)
    print(network.stimulations)
