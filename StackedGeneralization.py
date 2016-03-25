"""

Purpose: This script tries to implement a technique called stacking/blending/stacked generalization.
The reason I have to make this a runnable script because I found that there isn't really any
readable code that demonstrates this technique. You may find the pseudocode in various papers but they
are all each kind of different.

Author: Eric Chio "log0" <im.ckieric@gmail.com>

======================================================================================================
Summary:

Just to test an implementation of stacking. Using a cross-validated random forest and SVMs, I was
only able to achieve an accuracy of about 88% (with 1000 trees and up). Using stacked generalization 
I have seen a maximum of 93.5% accuracy. It does take runs to find it out though. This uses only 
(10, 20, 10) trees for the three classifiers.

This code is heavily inspired from the code shared by Emanuele (https://github.com/emanuele) , but I
have cleaned it up to makeit available for easy download and execution.

======================================================================================================
Methodology:

Three classifiers (RandomForestRegressor, ExtraTreesRegressor and a GradientBoostingRegressor
are built to be stacked by a LogisticRegression in the end.

Some terminologies first, since everyone has their own, I'll define mine to be clear:
- DEV SET, this is to be split into the training and validation data. It will be cross-validated.
- TEST SET, this is the unseen data to validate the generalization error of our final classifier. This
set will never be used to train.

======================================================================================================
Log Output:

X_test.shape = (62L, 6L)
blend_train.shape = (247L, 3L)
blend_test.shape = (62L, 3L)
Training classifier [0]
Fold [0]
Fold [1]
Fold [2]
Fold [3]
Fold [4]
Training classifier [1]
Fold [0]
Fold [1]
Fold [2]
Fold [3]
Fold [4]
Training classifier [2]
Fold [0]
Fold [1]
Fold [2]
Fold [3]
Fold [4]
Y_dev.shape = 247
Accuracy = 0.935483870968


"""

import numpy as np
from sklearn.naive_bayes import GaussianNB

np.random.seed(2016)

from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression, LinearRegression, BayesianRidge
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.utils import shuffle
from local_paths import INPUT_PATH
import xgboost as xgb


def run(X, Y):
    # The DEV SET will be used for all training and validation purposes
    # The TEST SET will never be used for training, it is the unseen set.

    Y = (Y - 1) / 2

    dev_cutoff = int(len(Y) * 2.5 / 5)

    X, Y = shuffle(X, Y)

    X_dev = X[:dev_cutoff]
    Y_dev = Y[:dev_cutoff]
    X_test = X[dev_cutoff:]
    Y_test = Y[dev_cutoff:]

    n_trees = 500  # Higher is better
    n_folds = 5  # Higher is better

    # Our level 0 classifiers
    clfs = [
        ('BayesianRidge', BayesianRidge(alpha_1=1e-6, alpha_2=1e-6)),
        ('LinearRegression', LinearRegression(n_jobs=-1)),
        ('RandomForestRegressor', RandomForestRegressor(n_estimators=n_trees, n_jobs=-1)),
        ('ExtraTreesRegressor', ExtraTreesRegressor(n_estimators=n_trees * 2, n_jobs=-1)),
        ('GradientBoostingRegressor', GradientBoostingRegressor(n_estimators=n_trees)),
        ('XGBLinear', xgb.XGBRegressor(learning_rate=0.05,
                                       # silent=False,
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
                                       n_estimators=n_trees * 2,
                                       max_depth=25
                                       )),
        ('XGBLogistic', xgb.XGBRegressor(learning_rate=0.05,
                                         # silent=False,
                                         objective="reg:logistic",
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
                                         n_estimators=n_trees * 2,
                                         max_depth=25
                                         ))
    ]

    # Ready for cross validation
    skf = list(StratifiedKFold(Y_dev, n_folds=n_folds, shuffle=True))

    # Pre-allocate the data
    blend_train = np.zeros((X_dev.shape[0], len(clfs)))  # Number of training data x Number of classifiers
    blend_test = np.zeros((X_test.shape[0], len(clfs)))  # Number of testing data x Number of classifiers

    print('X_test.shape = %s' % (str(X_test.shape)))
    print('blend_train.shape = %s' % (str(blend_train.shape)))
    print('blend_test.shape = %s' % (str(blend_test.shape)))

    # For each classifier, we train the number of fold times (=len(skf))
    for j, (clf_name, clf) in enumerate(clfs):
        print('Training classifier [%s %s]' % (j, clf_name))
        blend_test_j = np.zeros((X_test.shape[0], len(
            skf)))  # Number of testing data x Number of folds , we will take the mean of the predictions later
        for i, (train_index, cv_index) in enumerate(skf):
            print('Fold [%s]' % (i))

            # This is the training and validation set
            X_train = X_dev[train_index]
            Y_train = Y_dev[train_index]
            X_cv = X_dev[cv_index]
            Y_cv = Y_dev[cv_index]

            clf.fit(X_train, Y_train)

            # This output will be the basis for our blended classifier to train against,
            # which is also the output of our classifiers
            blend_train[cv_index, j] = clf.predict(X_cv)
            blend_test_j[:, i] = clf.predict(X_test)
        # Take the mean of the predictions of the cross validation set
        np.concatenate((X_dev, Y_dev.reshape((len(X_dev), 1)), blend_train[:, j].reshape((len(X_dev), 1))),
                                  axis=1).dump("pickle/" + clf_name + 'Train.numpy')
        blend_test[:, j] = blend_test_j.mean(1)

    print('Y_dev.shape = %s' % Y_dev.shape)

    # Start blending!
    bclf = LinearRegression(n_jobs=-1)

    # score=cross_val_score(bclf, verbose=20, cv=2, X=blend_train, y=Y_dev)
    bclf.fit(blend_train, Y_dev)

    # Predict now
    Y_test_predict = bclf.predict(blend_test)

    # Y_test_predict = 2*Y_test_predict+1

    for i in range(len(Y_test_predict)):
        if Y_test_predict[i] < 0:
            Y_test_predict[i] = 0
        elif Y_test_predict[i] > 1:
            Y_test_predict[i] = 1

    score = metrics.mean_squared_error(Y_test * 2 + 1, Y_test_predict * 2 + 1)
    print('Accuracy = %s' % (score ** 0.5))
    print('Weights = %s' % str(bclf.coef_))

    return score


if __name__ == '__main__':
    X_train = np.load(INPUT_PATH + 'X_train.numpy')
    # X_test = np.load(INPUT_PATH + 'X_test.numpy')
    y_train = np.load(INPUT_PATH + 'y_train.numpy')

    run(X_train, y_train)
