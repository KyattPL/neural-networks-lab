import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from keras import layers

BATCH_SIZE = 128
EPOCHS = 10
OUT_DIM = 16
EMBEDDING = 64
ACTIVATION = 'relu'
MAX_PAD = 100

# Load the data and split it between train and test sets
maxLens = [50, 80, 100, 130, 150]

for padding in maxLens:
    num_words = 15000
    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words = num_words)

    x_train = keras.preprocessing.sequence.pad_sequences(x_train, padding)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, padding)

    rnn = keras.Sequential(
        [
            layers.Embedding(num_words, EMBEDDING, input_length = len(x_train[0])),
            layers.SimpleRNN(OUT_DIM, input_shape = (num_words, padding), return_sequences=False, activation=ACTIVATION),
            layers.Dense(1),
            layers.Activation('sigmoid')
        ]
    )

    lstm = keras.Sequential(
        [
            layers.Embedding(num_words, EMBEDDING, input_length=len(x_train[0])),
            layers.LSTM(OUT_DIM, input_shape=(num_words, padding), return_sequences=False, activation=ACTIVATION),
            layers.Dense(1),
            layers.Activation('sigmoid')
        ]
    )

    rnn.compile(loss="binary_crossentropy",
                    optimizer="rmsprop", metrics=["accuracy"])

    lstm.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

    model_history = rnn.fit(x_train, y_train, batch_size=BATCH_SIZE,
                                epochs=EPOCHS, validation_data=(x_test, y_test))

    lstm_history = lstm.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_test, y_test))

    score = rnn.evaluate(x_test, y_test, verbose=0)
    score_lstm = lstm.evaluate(x_test, y_test, verbose=0)

    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    print("Test loss:", score_lstm[0])
    print("Test accuracy:", score_lstm[1])

    plt.plot(model_history.history['accuracy'])
    plt.plot(model_history.history['val_accuracy'])
    plt.title('Dokładność modelu - RNN')
    plt.ylabel('precyzja')
    plt.xlabel('epoki')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f'plots/RNN-acc-padding-{padding}.png')

    plt.clf()

    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('Strata modelu - RNN')
    plt.ylabel('strata')
    plt.xlabel('epoki')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f'plots/RNN-loss-padding-{padding}.png')

    plt.clf()

    plt.plot(lstm_history.history['accuracy'])
    plt.plot(lstm_history.history['val_accuracy'])
    plt.title('Dokładność modelu - LSTM')
    plt.ylabel('precyzja')
    plt.xlabel('epoki')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f'plots/LSTM-acc-padding-{padding}.png')

    plt.clf()

    plt.plot(lstm_history.history['loss'])
    plt.plot(lstm_history.history['val_loss'])
    plt.title('Strata modelu - LSTM')
    plt.ylabel('strata')
    plt.xlabel('epoki')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f'plots/LSTM-loss-padding-{padding}.png')

    plt.clf()

    y_pred = rnn.predict(x_test)
    y_pred = (y_pred > 0.5)
    cm = confusion_matrix(y_test, y_pred)

    df1 = pd.DataFrame(columns=["True","False"], index= ["True","False"], data=cm ) 

    f,ax = plt.subplots(figsize=(3,3))
    sns.heatmap(df1, annot=True,cmap="Blues", fmt= '.0f',ax=ax)
    plt.xlabel("Predykcja etykiety")
    plt.xticks(size = 12)
    plt.yticks(size = 12, rotation = 0)
    plt.ylabel("Prawdziwa etykieta")
    plt.title("Macierz pomyłek", size = 12)
    plt.savefig(f'plots/RNN-conf-matrix-padding-{padding}.png')
    plt.clf()

    y_pred = lstm.predict(x_test)
    y_pred = (y_pred > 0.5)
    cm = confusion_matrix(y_test, y_pred)

    df1 = pd.DataFrame(columns=["True","False"], index= ["True","False"], data=cm ) 

    f,ax = plt.subplots(figsize=(3,3))
    sns.heatmap(df1, annot=True,cmap="Blues", fmt= '.0f',ax=ax)
    plt.xlabel("Predykcja etykiety")
    plt.xticks(size = 12)
    plt.yticks(size = 12, rotation = 0)
    plt.ylabel("Prawdziwa etykieta")
    plt.title("Macierz pomyłek", size = 12)
    plt.savefig(f'plots/LSTM-conf-matrix-padding-{padding}.png')
    plt.clf()
