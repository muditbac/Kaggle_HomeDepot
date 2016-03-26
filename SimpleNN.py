from keras.callbacks import EarlyStopping
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras.optimizers import SGD

from sklearn.base import BaseEstimator

__author__ = 'mudit'

stop = EarlyStopping(monitor='loss', patience=2, verbose=0, mode='auto')


class KerasNN(BaseEstimator):
    def __init__(self, nb_epoch=64, batch_size=64, verbose=False):
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, X, y, **kwargs):

        if hasattr(self, 'model'):
            delattr(self, 'model')

        self.model = Sequential()

        input_dim = X.shape[1]

        self.model.add(Dense(64, input_dim=input_dim ),)
        self.model.add(Activation('tanh'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, init='normal'))
        self.model.add(Activation('tanh'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, init='normal'))
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss='mse', optimizer='adadelta')

        self.model.fit(X, y,
                       nb_epoch=self.nb_epoch,
                       batch_size=self.batch_size,
                       show_accuracy=self.verbose, **kwargs)

    def predict(self, X):
        return self.model.predict(X).reshape((X.shape[0]))
