from utilities import RMSE

__author__ = 'mudit'

import time

start_time = time.time()

import pandas as pd
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from SimpleNN import KerasNN
import xgboost as xgb
from configs import *
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from utilities import change_to_int
# random.seed(2016)

model_name = 'knn'
# Available options ['tree', 'rfr', 'xgb', 'xtree', 'gbr', 'knn']

# Choosing parameters on subsets.
X_train = np.load(INPUT_PATH + 'X_train.numpy')
# X_test = np.load(INPUT_PATH + 'X_test.numpy')
y_train = np.load(INPUT_PATH + 'y_train.numpy')
# id_test = np.load(INPUT_PATH + 'id_test.numpy')

y_train = (y_train - 1.0) / 2.0

X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.5, random_state=configs['seed'])

if model_name == 'rfr':
    level0 = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=2016, verbose=1)
    param_grid = {'max_features': [25], 'max_depth': [25]}
    h_param_grid = {'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
                    'max_features': hp.quniform('n_estimators', 20, 100, 1),
                    }
elif model_name == 'xgb':
    level0 = xgb.XGBRegressor(learning_rate=0.05,
                              silent=False,
                              objective="reg:linear",
                              nthread=-1,
                              gamma=0.5,
                              min_child_weight=5,
                              max_delta_step=1,
                              subsample=0.7,
                              colsample_bytree=0.7,
                              colsample_bylevel=1,
                              reg_alpha=0.5,
                              reg_lambda=1,
                              scale_pos_weight=1,
                              base_score=0.5,
                              seed=0,
                              missing=None,
                              n_estimators=10)
    h_param_grid = {'max_depth': hp.quniform('max_depth', 1, 13, 1),
                    'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
                    'learning_rate': hp.quniform('learning+rate', 0.025, 0.5, 0.025),
                    'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
                    'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
                    'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
                    }
    # {'max_depth': 7.0, 'learning+rate': 0.075000000000000011, 'gamma': 0.55000000000000004, 'colsample_bytree': 0.90000000000000002, 'n_estimators': 263.0, 'subsample': 0.65000000000000002} : 0.46..
elif model_name == 'tree':
    level0 = DecisionTreeRegressor(random_state=2016)
    h_param_grid = {'max_features': hp.quniform('max_features', 50, 100, 1),
                    'max_depth': hp.quniform('max_depth', 1, 15, 1)}
    # {'max_features': 95.0, 'max_depth': 6.0} : 0.47...
elif model_name == 'xtree':
    level0 = ExtraTreesRegressor(random_state=2016, n_jobs=-1)
    h_param_grid = {'max_features': hp.quniform('max_features', 50, 100, 1),
                    'max_depth': hp.quniform('max_depth', 5, 7, 1),
                    'n_estimators': hp.quniform('n_estimators', 100, 1000, 1)}
    # {'max_depth': 7.0, 'n_estimators': 478.0, 'max_features': 99.0}
elif model_name == 'gbr':
    level0 = GradientBoostingRegressor(random_state=2016, verbose=20)
    h_param_grid = {'max_depth': hp.quniform('max_depth', 1, 13, 1),
                    'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
                    'learning_rate': hp.quniform('learning+rate', 0.025, 0.5, 0.025),
                    'max_features': hp.quniform('max_features', 50, 100, 1),
                    'n_estimators': hp.quniform('n_estimators', 50, 500, 1),
                    }
elif model_name == 'knn':
    level0 = KerasNN()
    h_param_grid = {'d1': hp.loguniform('d1', -4, 0.1),
                    'd2': hp.loguniform('d2', -4, 0.1),
                    'lr': hp.quniform('lr', 0, 1, 0.1)
                    }


# Grid Search Implementation
# model = GridSearchCV(estimator=level0, param_grid=param_grid, n_jobs=NJOBS, cv=2, verbose=20, scoring=RMSE, refit=False)
#
# model.fit(X_train, y_train)
#
# print("Best parameters found by grid search:")
# print(model.best_params_)
# print("Best CV score:")
# print(model.best_score_)

# Hyperopt Implementatation
def score(params):
    # change_to_int(params, ['max_depth', 'n_estimators'])
    # change_to_int(params, ['max_depth', 'max_features', 'n_estimators'])
    print('Parameters :')
    print(params)
    # params['max_depth'] = int(params['max_depth'])
    level0.set_params(**params)
    level0.fit(X_train, y_train)
    y_pred = level0.predict(X_test)
    score = mean_squared_error(y_test, y_pred)
    score = score ** 0.5
    score = score * 2
    print('Score %f' % score)
    return {'loss': score, 'status': STATUS_OK}


def optimize(trials):
    best = fmin(score, h_param_grid, algo=tpe.suggest, trials=trials, max_evals=20)
    print(best)
    return best


trials = Trials()
optimize(trials)
