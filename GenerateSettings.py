import  os
from configs import *
from sklearn.cross_validation import StratifiedKFold
import pandas as pd
import pickle as pkl
__author__ = 'mudit'


paths = [PROCESS_PATH, FOLD_PATH]

for i in range(configs['n_folds']):
    paths.append(FOLD_PATH + 'fold' + str(i+1))

for path in paths:
    if not os.path.exists(path):
        os.makedirs(path)


train = pd.read_csv(TRAIN_FILE, encoding="ISO-8859-1")
y_train = train['relevance'].values

folds = list(StratifiedKFold(y_train, n_folds=configs['n_folds'], shuffle=True))

pkl.dump(folds, open(FOLDS_INFO, 'wb'))

for i, each_fold in enumerate(folds):
    pkl.dump(each_fold, open(FOLD_PATH + 'fold' + str(i+1) + '/fold.pkl', 'wb'))