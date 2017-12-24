import numpy as np

import keras
from keras.models import Sequential
from keras.layers import SimpleRNN
from keras.layers import Dense, Dropout

class EventModel(object):
    
    def __init__(self, D):
        self.D = D

    def is_recurrent(self):
        return self.isRecurrent
        
    def predict(self, X):
        return np.copy(X)

class BasicMLP(EventModel):

    # layers[i] is (dimensionality, activation) for layer i, where the first hidden layer is layer 0
    # dropouts[i] is the dropout before layer i
    def __init__(self, D, layers = None, dropouts = None):
        # initialize default layers and dropouts
        if layers == None:
            layers = [(D, 'linear')]
        if dropouts == None:
            dropouts = len(layers) * [0.0]
        if len(layers) < 1:
            raise ValueError('invalid number of layers') 
        if layers[-1][0] != D:
            raise ValueError('dimensionality of input and output must match')        
    
        model = Sequential()
        if dropouts[0] > 0:
            model.add(Dropout(dropouts[0], input_shape=(D,)))
            model.add(Dense(units=layers[0][0], activation=layers[0][1]))
        else:
            model.add(Dense(units=layers[0][0], input_dim = D, activation=layers[0][1]))

        for i in range(1, len(layers)):
            if dropouts[i] > 0:
                model.add(Dropout(dropouts[i]))
            model.add(Dense(units=layers[i][0], activation=layers[i][1]))
        model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

        self.model = model
        self.isRecurrent = False
            
    def update(self, X, Y):
        self.model.fit(np.asarray(X), np.asarray(Y), epochs=1, batch_size=1, verbose=0)
        
    def predict(self, X):
        return self.model.predict(np.asarray(X))

class BasicRNN(EventModel):

    # layers[i] is (dimensionality, activation) for layer i, where the first hidden layer is layer 0
    # dropouts[i] is dropout before layer i
    def __init__(self, D, layers = None, dropouts = None):
        # initialize default layers and dropouts
        if layers == None:
            layers = [(D, 'linear')]
        if dropouts == None:
            dropouts = len(layers) * [0.0]
        if len(layers) < 1:
            raise ValueError('invalid number of layers') 
        if layers[-1][0] != D:
            raise ValueError('dimensionality of input and output must match')        

        model = Sequential()
        model.add(SimpleRNN(units=layers[0][0], input_shape=(None, D), return_sequences=True, activation=layers[0][1], dropout=dropouts[0]))
        for i in range(1, len(layers) - 1):
            model.add(SimpleRNN(units=layers[i][0], return_sequences=True, activation=layers[i][1], dropout=dropouts[0]))

        if dropouts[-1] > 0:
            model.add(Dropout(dropouts[-1]))
        model.add(Dense(units=layers[-1][0], activation=layers[-1][1]))

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