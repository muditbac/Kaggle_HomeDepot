__author__ = 'mudit'

import numpy as np
from configs import FEATURES_PATH
import os

def generate_dataset(features, filename):
    pass


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