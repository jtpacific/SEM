import keras
from keras.models import Sequential
from keras.layers import SimpleRNN, LSTM
import numpy as np
from keras.layers import Dense, Dropout

# initialize simple rnn
def init(batches, seq_length, vector_length):
    model = Sequential()
    
    model.add(SimpleRNN(units=20, input_shape=(seq_length, vector_length), return_sequences=True))
    model.add(SimpleRNN(units=20, return_sequences=True))
    model.add(SimpleRNN(units=vector_length, activation='linear', return_sequences=True))

    model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
    return model

# predict the next scene given a sequence of scenes
def predict(scenes, model, ret_all = False):
    predictions = model.predict(np.asarray([scenes]))
    if ret_all:
        return np.asarray(predictions[0])
    return np.asarray(predictions[0][-1])

# train on a sequence of scenes
def train(scenes, model):
    trainX = scenes[0:len(scenes) - 1]
    trainY = scenes[1:len(scenes)]
    model.fit(np.asarray([trainX]), np.asarray([trainY]), epochs=1, batch_size=1)