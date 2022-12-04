import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from keras import layers

NUM_NEURONS_FIRST = 128
NUM_NEURONS_OUT = 10
NUM_FILTERS = 32
KERNEL_X = 3
KERNEL_Y = 3
KERNEL_STRIDE = 1
POOL_X = 2
POOL_Y = 2
POOL_STRIDE = 2
BATCH_SIZE = 128
EPOCHS = 10
VALIDATION_PERCENT = 0.1

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

batches = [5, 20, 100, 500, 1000]
# names = ['MAX', 'AVG', 'GMAX', 'GAVG']
# pools = [layers.MaxPooling2D(pool_size=(POOL_X, POOL_Y), strides=POOL_STRIDE),
#         layers.AveragePooling2D(pool_size=(POOL_X, POOL_Y), strides=POOL_STRIDE),
#         layers.GlobalMaxPooling2D(),
#         layers.GlobalAveragePooling2D()]

# dropout_rates = [0.01, 0.05, 0.1, 0.2, 0.4]

# i = 0
for curr_batch in batches:

    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    mnist = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Flatten(),
            layers.Dense(NUM_NEURONS_FIRST, activation="relu"),
            layers.Dense(NUM_NEURONS_OUT, activation="softmax")
        ]
    )

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(NUM_FILTERS, kernel_size=(KERNEL_X, KERNEL_Y), activation="relu", strides=KERNEL_STRIDE),
            layers.MaxPooling2D(pool_size=(POOL_X, POOL_Y), strides=POOL_STRIDE),
            layers.Flatten(),
            layers.Dense(num_classes, activation="softmax")
        ]
    )

    batch_size = curr_batch
    epochs = EPOCHS

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    mnist.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model_history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=VALIDATION_PERCENT)
    mnist_history = mnist.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=VALIDATION_PERCENT)

    score = model.evaluate(x_test, y_test, verbose=0)
    mnist_score = mnist.evaluate(x_test, y_test, verbose=0)

    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    print("Test loss:", mnist_score[0])
    print("Test accuracy:", mnist_score[1])

    plt.plot(model_history.history['accuracy'])
    plt.plot(model_history.history['val_accuracy'])
    plt.title('Dokładność modelu - CNN')
    plt.ylabel('precyzja')
    plt.xlabel('epoki')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f'plots/CNN-acc-batch-{curr_batch}.png')

    plt.clf()

    plt.plot(mnist_history.history['accuracy'])
    plt.plot(mnist_history.history['val_accuracy'])
    plt.title('Dokładność modelu - MLP')
    plt.ylabel('precyzja')
    plt.xlabel('epoki')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f'plots/MLP-acc-batch-{curr_batch}.png')

    plt.clf()

    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('Strata modelu - CNN')
    plt.ylabel('strata')
    plt.xlabel('epoki')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f'plots/CNN-loss-batch-{curr_batch}.png')

    plt.clf()

    # i += 1
    plt.plot(mnist_history.history['loss'])
    plt.plot(mnist_history.history['val_loss'])
    plt.title('Strata modelu - MLP')
    plt.ylabel('strata')
    plt.xlabel('epoki')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f'plots/MLP-loss-batch-{curr_batch}.png')

    plt.clf()