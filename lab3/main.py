from MLP import MLP

#     O  O
#    /\ /\
#   O  O  O
#  / \ /\ /\
# O  O  O  O


if __name__ == "__main__":
    testInp = [0, 1, 1, 0]

    network = MLP(3, [4, 3, 2], [None, None], 0.1)
    o = network.calc_outputs(testInp)
    print(o)
