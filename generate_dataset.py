__author__ = 'mudit'

import numpy as np
from configs import FEATURES_PATH, LEN_TRAIN, DATASET_PATH
import os


def generate_dataset(features, dataset_name):

    final = None
    for i, (feature, transformer) in enumerate(features):

        x_all = np.load('%s%s.npy' % (FEATURES_PATH, feature))
        x_all = transformer.fit_transform(x_all)

        if len(x_all.shape) == 1:
            x_all.shape = (x_all.shape[0], 1)

        if final is None:
            assert i == 0
            final = x_all
        else:
            final = np.hstack((final, x_all))

    X_train = final[:LEN_TRAIN]
    X_test = final[LEN_TRAIN:]

    X_train.dump('%s%s_train.npy' % (DATASET_PATH, dataset_name))
    X_test.dump('%s%s_test.npy' % (DATASET_PATH, dataset_name))

def save_feature(data, features_name):
    if not isinstance(data, np.ndarray):
        data = data.values

    data.dump(FEATURES_PATH + features_name + '.npy')


def save_dataset(data, dataset):
    pass


def generate_features_list():
    for filename in os.listdir(FEATURES_PATH):
        tokens = filename.split('.')
        if tokens[-1] == 'npy':
            print(tokens[0])
