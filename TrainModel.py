from configs import INPUT_PATH
from utils import RMSE

__author__ = 'mudit'

import time

start_time = time.time()

import pandas as pd
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from SimpleNN import KerasNN
import xgboost as xgb

# random.seed(2016)

model_name = 'knn'
# Available options ['tree', 'rfr', 'xgb', 'xtree', 'gbr', 'knn']

X_train = np.load(INPUT_PATH + 'X_train.numpy')
X_test = np.load(INPUT_PATH + 'X_test.numpy')
y_train = np.load(INPUT_PATH + 'y_train.numpy')
id_test = np.load(INPUT_PATH + 'id_test.numpy')

y_train = (y_train - 1.0)/2.0

if model_name == 'rfr':
    level0 = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=2016, verbose=1)
    param_grid = {'model0__max_features': [25], 'model0__max_depth': [25]}

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
                              missing=None)
    param_grid = {'n_estimators': [2000], 'max_depth': [10]}
elif model_name == 'tree':
    level0 = DecisionTreeRegressor(random_state=2016)
    param_grid = {'max_features': [25], 'max_depth': [25]}
elif model_name == 'xtree':
    level0 = ExtraTreesRegressor(random_state=2016, n_jobs=-1)
    param_grid = {'max_features': [25], 'max_depth': [25]}
elif model_name == 'gbr':
    level0 = GradientBoostingRegressor(random_state=2016, verbose=20)
    param_grid = {'max_features': [25], 'max_depth': [25], 'n_estimators': [20]}
elif model_name == 'knn':
    level0 = KerasNN(nb_epoch=16)
    param_grid = {'nb_epoch': [8,16,32]}

model = GridSearchCV(estimator=level0, param_grid=param_grid, n_jobs=1, cv=2, verbose=20, scoring=RMSE)

model.fit(X_train, y_train)

print("Best parameters found by grid search:")
print(model.best_params_)
print("Best CV score:")
print(model.best_score_)

y_pred = model.predict(X_test)

y_pred = y_pred*2+1

for i in range(len(y_pred)):
    if y_pred[i] < 1.0:
        y_pred[i] = 1.0
    if y_pred[i] > 3.0:
        y_pred[i] = 3.0

# y_pred = np.round(y_pred*3)/3.0

pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission/submission_%s_25.csv'%model_name, index=False)
print("--- Training & Testing: %s minutes ---" % round(((time.time() - start_time) / 60), 2))
