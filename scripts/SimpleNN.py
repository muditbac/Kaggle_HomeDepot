from keras.callbacks import EarlyStopping
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras.optimizers import Adadelta, SGD
from keras.regularizers import l2

from sklearn.base import BaseEstimator

__author__ = 'mudit'

stop = EarlyStopping(monitor='loss', patience=2, verbose=0, mode='auto')


class KerasNN(BaseEstimator):
    def __init__(self, nb_epoch=50, batch_size=512, d1=0.5, d2=0.5, lr=1, verbose=False):
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.verbose = verbose
        self.d1 = d1
        self.d2 = d2
        self.lr = lr

    def fit(self, X, y, **kwargs):
        if hasattr(self, 'model'):
            delattr(self, 'model')

        self.model = Sequential()

        input_dim = X.shape[1]
        print(self.d1)
        print(self.d2)

        self.model.add(Dense(500, input_dim=input_dim, W_regularizer=l2(self.d1)), )
        self.model.add(Activation('sigmoid'))
        # self.model.add(Dropout(self.d1))
        self.model.add(Dense(500, W_regularizer=l2(self.d2)))
        self.model.add(Activation('sigmoid'))
        # self.model.add(Dropout(self.d2))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss='mse', optimizer=Adadelta(lr=self.lr))

        self.model.fit(X, y,
                       nb_epoch=self.nb_epoch,
                       batch_size=self.batch_size,
                       show_accuracy=self.verbose, **kwargs)

    def predict(self, X):
        return self.model.predict_proba(X).reshape((X.shape[0]))
