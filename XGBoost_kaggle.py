import random
import numpy as np
import time

np.random.seed(2016)
random.seed(2016)

from configs import INPUT_PATH

import pandas as pd
from sklearn import pipeline  # model_selection
from sklearn.grid_search import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, make_scorer
from nltk.stem.porter import *

__author__ = 'mudit'
start_time = time.time()

stemmer = PorterStemmer()
import xgboost as xgb


class cust_regression_vals(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, hd_searches):
        d_col_drops = ['id', 'relevance', 'search_term', 'product_title', 'product_description', 'product_info', 'attr',
                       'brand']
        hd_searches = hd_searches.drop(d_col_drops, axis=1).values
        return hd_searches


class cust_txt_col(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key].apply(str)


def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions) ** 0.5
    return fmean_squared_error_


RMSE = make_scorer(fmean_squared_error, greater_is_better=False)

df_test = pd.read_csv(INPUT_PATH + 'df_test2.csv')
df_train = pd.read_csv(INPUT_PATH+ 'df_train2.csv')

id_test = df_test['id']
y_train = df_train['relevance'].values
X_train = df_train[:].fillna(0)
X_test = df_test[:].fillna(0)

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
tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
tsvd = TruncatedSVD(n_components=20, random_state=2016)
tnmf = NMF(n_components=20, random_state=2016)
clf = pipeline.Pipeline([
    ('union', FeatureUnion(
        transformer_list=[
            ('cst', cust_regression_vals()),
            ('txt1', pipeline.Pipeline([('s1', cust_txt_col(key='search_term')), ('tfidf1', tfidf), ('tsvd1', tsvd)])),
            (
                'txt2',
                pipeline.Pipeline([('s2', cust_txt_col(key='product_title')), ('tfidf2', tfidf), ('tsvd2', tsvd)])),
            ('txt3',
             pipeline.Pipeline([('s3', cust_txt_col(key='product_description')), ('tfidf3', tfidf), ('tsvd3', tsvd)])),
            ('txt4', pipeline.Pipeline([('s4', cust_txt_col(key='brand')), ('tfidf4', tfidf), ('tsvd4', tsvd)]))
        ],
        transformer_weights={
            'cst': 1.0,
            'txt1': 0.5,
            'txt2': 0.25,
            'txt3': 0.0,
            'txt4': 0.5
        },
        # n_jobs = -1
    )),
    ('xgb_model', xgb_model)])
param_grid = {'xgb_model__n_estimators': [2000], 'xgb_model__max_depth': [10]}
model = GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=-1, cv=2, verbose=20, scoring=RMSE)

model.fit(X_train, (y_train-1)/2)

print("Best parameters found by grid search:")
print(model.best_params_)
print("Best CV score:")
print(model.best_score_)

y_pred = model.predict(X_test)

for i in range(len(y_pred)):
    if y_pred[i]<1.0:
        y_pred[i] = 1.0
    if y_pred[i]>3.0:
        y_pred[i] = 3.0

# y_pred = np.round(y_pred*3)/3.0

pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission_spell_corrected.csv', index=False)
print("--- Training & Testing: %s minutes ---" % round(((time.time() - start_time) / 60), 2))
"""
New features set without n-gram

Best parameters found by grid search:
{'xgb_model__max_depth': 10, 'xgb_model__n_estimators': 2000}
Best CV score:
-0.469944452368
--- Training & Testing: 57.47 minutes ---

with n gram and spelling corrected
Best CV score:
-0.469065242165


On 1000 entries

reg:linear 0.521644
reg:logistic 0.549528

reg:linear 0.52188 (Seed 2016)
reg:linear 0.57520 alpha = 0.5
"""