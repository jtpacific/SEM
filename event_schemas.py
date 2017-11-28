import numpy as np

import keras
from keras.models import Sequential
from keras.layers import SimpleRNN, LSTM
from keras.layers import Dense, Dropout

class EventModel(object):
    
    def __init__(self, D):
        self.D = D

    def is_recurrent(self):
        return self.isRecurrent
        
    def predict(self, X):
        return np.copy(X)

class LinearMLP(EventModel):

    def __init__(self, D):
        model = Sequential()
        model.add(Dense(units=10, input_dim=D, activation='linear'))
        model.add(Dense(units=D, activation='linear'))
        model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
        self.model = model
        self.isRecurrent = False
            
    def update(self, X, Y):
        self.model.fit(np.asarray(X), np.asarray(Y), epochs=1, batch_size=1, verbose=0)
        
    def predict(self, X, ret_all = False):
        predictions = self.model.predict(np.asarray(X))
        if ret_all:
            return np.asarray(predictions[0])
        return np.asarray(predictions[0][-1])

class LinearRNN(EventModel):

    def __init__(self, D):
        model = Sequential()
        model.add(SimpleRNN(units=10, input_shape=(None, D), activation='linear', return_sequences=True))
        model.add(SimpleRNN(units=D, activation='linear', return_sequences=True))
        model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
        self.model = model
        self.isRecurrent = True
        
    def train_recurrent(self, scenes):
        if len(scenes) < 2:
            return
        trainX = scenes[0:len(scenes) - 1]
        trainY = scenes[1:len(scenes)]
        self.model.fit(np.asarray([trainX]), np.asarray([trainY]), epochs=1, batch_size=1, verbose=0)
        
    def predict(self, X, ret_all = False):
        predictions = self.model.predict(np.asarray([X]))
        if ret_all:
            return np.asarray(predictions[0])
        return np.asarray(predictions[0][-1])

class BasicRNN(EventModel):

    def __init__(self, D, hidden, dropout=0.5):
        model = Sequential()
        model.add(SimpleRNN(units=hidden, input_shape=(None, D), return_sequences=True, dropout=dropout))
        model.add(SimpleRNN(units=hidden, return_sequences=True))
        model.add(SimpleRNN(units=D, activation='linear', return_sequences=True))
        model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
        self.model = model
        self.isRecurrent = True
        
    def train_recurrent(self, scenes):
        if len(scenes) < 2:
            return
        trainX = scenes[0:len(scenes) - 1]
        trainY = scenes[1:len(scenes)]
        self.model.fit(np.asarray([trainX]), np.asarray([trainY]), epochs=1, batch_size=1, verbose=0)
        
    def predict(self, X, ret_all = False):
        predictions = self.model.predict(np.asarray([X]))
        if ret_all:
            return np.asarray(predictions[0])
        return np.asarray(predictions[0][-1])

class BatchRNN(EventModel):

    def __init__(self, D, hidden):
        model = Sequential()
        model.add(SimpleRNN(units=hidden, input_shape=(None, D), return_sequences=True, batch_size=3))
        model.add(SimpleRNN(units=hidden, return_sequences=True))
        model.add(SimpleRNN(units=D, activation='linear', return_sequences=True))
        model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
        self.model = model
        self.isRecurrent = True
        
    def train_recurrent(self, scenes):
        if len(scenes) < 2:
            return
        trainX = scenes[0:len(scenes) - 1]
        trainY = scenes[1:len(scenes)]
        self.model.fit(np.asarray([trainX]), np.asarray([trainY]), epochs=1, batch_size=1, verbose=0)
        
    def predict(self, X, ret_all = False):
        predictions = self.model.predict(np.asarray([X]))
        if ret_all:
            return np.asarray(predictions[0])
        return np.asarray(predictions[0][-1])

class BoundedRNN(EventModel):

    def __init__(self, D):
        model = Sequential()
        model.add(SimpleRNN(units=10, input_shape=(None, D), return_sequences=True))
        model.add(SimpleRNN(units=D, activation='tanh', return_sequences=True))
        model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
        self.model = model
        self.isRecurrent = True
        
    def train_recurrent(self, scenes):
        if len(scenes) < 2:
            return
        trainX = scenes[0:len(scenes) - 1]
        trainY = scenes[1:len(scenes)]
        self.model.fit(np.asarray([trainX]), np.asarray([trainY]), epochs=1, batch_size=1, verbose=0)
        
    def predict(self, X, ret_all = False):
        predictions = self.model.predict(np.asarray([X]))
        if ret_all:
            return np.asarray(predictions[0])
        return np.asarray(predictions[0][-1])