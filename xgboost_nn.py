from configs import OUTPUT_PATH

__author__ = 'mudit'

import xgboost as xgb
import pandas as pd
xgb_model = xgb.XGBRegressor(learning_rate=0.05,
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

train_X = pd.read_csv(OUTPUT_PATH + 'X_features.csv')

train_y = train_X['relevance'].values
train_X = train_X.drop(['relevance'], axis=1).values

xgb_model.max_depth = 10
xgb_model.n_estimators = 2000

xgb_model.fit(train_X, train_y)

# Getting test results
test_Y = pd.read_csv(OUTPUT_PATH + 'X_features_test.csv').values

y_pred = xgb_model.predict(test_Y)

for i in range(len(y_pred)):
    if y_pred[i]<1.0:
        y_pred[i] = 1.0
    if y_pred[i]>3.0:
        y_pred[i] = 3.0

# TODO Load df_test
pd.DataFrame({"id": df_test['id'].values, "relevance": y_pred}).to_csv(OUTPUT_PATH + 'submission_nn_xgb.csv', index=False)