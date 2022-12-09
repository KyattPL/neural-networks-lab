import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from keras import layers

# Load the data and split it between train and test sets
num_words = 15000
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words = num_words)

maxlen = 130
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen)


rnn = keras.Sequential(
    [
        layers.Embedding(num_words, 32, input_length = len(x_train[0])),
        layers.SimpleRNN(16, input_shape = (num_words, maxlen), return_sequences=False, activation='relu'),
        layers.Dense(1),
        layers.Activation('sigmoid')
    ]
)

batch_size = 128
epochs = 10

rnn.compile(loss="binary_crossentropy",
                optimizer="rmsprop", metrics=["accuracy"])

model_history = rnn.fit(x_train, y_train, batch_size=batch_size,
                            epochs=epochs, validation_data=(x_test, y_test))

score = rnn.evaluate(x_test, y_test, verbose=0)

print("Test loss:", score[0])
print("Test accuracy:", score[1])

plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('Dokładność modelu - CNN')
plt.ylabel('precyzja')
plt.xlabel('epoki')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(f'plots/RNN-acc.png')

plt.clf()

plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('Strata modelu - CNN')
plt.ylabel('strata')
plt.xlabel('epoki')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(f'plots/RNN-loss.png')

plt.clf()

y_pred = rnn.predict(x_test)
y_pred = (y_pred > 0.5)
cm = confusion_matrix(y_test, y_pred)

df1 = pd.DataFrame(columns=["True","False"], index= ["True","False"], data= cm ) 

f,ax = plt.subplots(figsize=(3,3))
sns.heatmap(df1, annot=True,cmap="Blues", fmt= '.0f',ax=ax)
plt.xlabel("Predykcja etykiety")
plt.xticks(size = 12)
plt.yticks(size = 12, rotation = 0)
plt.ylabel("Prawdziwa etykieta")
plt.title("Macierz pomyłek", size = 14)
plt.savefig(f'plots/RNN-conf-matrix.png')
plt.clf()
