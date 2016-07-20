from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
from utilities import make_folder
from configs import *
import pickle as pkl
import pandas as pd

__author__ = 'mudit'

paths = [PROCESS_PATH, FOLD_PATH, FEATURES_PATH, DATASET_PATH]

for i in range(configs['n_folds']):
    paths.append(FOLD_PATH + 'fold' + str(i + 1))

for path in paths:
    make_folder(path)

train = pd.read_csv(TRAIN_FILE, encoding="ISO-8859-1")
y_train = train['relevance'].values

if not DEBUG:
    folds = list(StratifiedKFold(y_train, n_folds=configs['n_folds'], shuffle=True))
    pkl.dump(folds, open(FOLD_PATH + 'folds.pkl', 'wb'))
else:
    split = list(StratifiedShuffleSplit(y_train, test_size=(1 - TEST_CUTOFF), n_iter=1))[0]
    pkl.dump(split, open(FOLD_PATH + 'split.pkl', 'wb'))

    y = y_train[split[0]]
    folds = list(StratifiedKFold(y, n_folds=configs['n_folds'], shuffle=True))
    pkl.dump(folds, open(FOLD_PATH + 'split_folds.pkl', 'wb'))