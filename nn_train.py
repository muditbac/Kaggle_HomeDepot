import pickle
import logging

import pandas as pd
import numpy as np

np.random.seed(2016)

from keras.layers import Embedding, Flatten, Dense
from keras.models import Graph
from keras.regularizers import ActivityRegularizer, WeightRegularizer, l2
from keras.callbacks import ModelCheckpoint, Callback

from configs import *

__author__ = 'mudit'

logging.getLogger().setLevel(logging.INFO)
[product_uid, output, train_query_vectors, query_vectors, product_vectors, prod] = pickle.load(
    open(OUTPUT_PATH + 'nn.pickle', 'rb'))


# output = np.load(DATASET_PATH + 'Y_train.npy')
# df_train = pd.read_csv(INPUT_PATH + 'df_train2.csv')
# df_test = pd.read_csv(INPUT_PATH + 'df_test2.csv')

# len_train = len(df_train)

# input_features = pd.DataFrame(input_features).fillna(0).values

# cols = ['len_of_query', 'len_of_title',
#         'len_of_description', 'len_of_brand',
#         'query_in_title', 'query_in_description', 'edit_dist_in_info',
#         'edit_ratio_in_info', 'word_in_title', 'edit_in_title',
#         'seq_edit_in_title', 'word_in_description', 'word_in_brand',
#         'ratio_title', 'ratio_description', 'ratio_brand', 'ngram_match_title',
#         'ngram_match_description', 'brand_feature', 'search_term_feature']

# input_features = df_train[cols].fillna(0).values

# product_uid = df_train['product_uid'].values.reshape((1000, 1))

# train_query_vectors = train_query_vectors[:1000]


class SelectBestValidation(Callback):
    def on_train_begin(self, logs={}):
        self.best_validation_score = -1

    def on_epoch_end(self, epoch, logs={}):
        val_score = logs.get('val_loss')
        if self.best_validation_score == -1 or val_score <= self.best_validation_score:
            self.best_validation_score = val_score
            self.weights = graph.get_weights()
            print('Saving model for val_score = %f' % val_score)


# product_uid = df_train['product_uid'].map(lambda x: product_id[x]).values.reshape(len_train, 1)
# input_features = df_train[['len_of_query', 'len_of_title',
#                            'len_of_description', 'len_of_brand',
#                            'query_in_title', 'query_in_description', 'word_in_title',
#                            'word_in_description', 'word_in_brand', 'ratio_title',
#                            'ratio_description', 'ratio_brand', 'brand_feature',
#                            'search_term_feature']].values
# output = df_train['relevance'].values
# train_query_vectors = query_vectors[:len_train]
r = 0
graph = Graph()
# graph.add_input(name='input_features', input_shape=(input_features.shape[1],))
graph.add_input(name='product_uid', input_shape=(1,), dtype=int)
graph.add_input(name='query_vector', input_shape=(50,))
graph.add_node(Embedding(input_dim=len(prod) + 1,
                         output_dim=100,
                         weights=[np.concatenate((np.zeros((1, 100)), product_vectors), axis=0)],
                         trainable=True,
                         input_length=1,
                         W_regularizer=l2(r)),
               name='embedding',
               input='product_uid', )
graph.add_node(Flatten(), name='flatten', input='embedding')
graph.add_node(Dense(50, activation='sigmoid', W_regularizer=l2(r)), name='hidden0',
               inputs=['query_vector', 'flatten'],
               merge_mode="concat", concat_axis=1)
# graph.add_node(Dense(50, activation='sigmoid'), name='hidden1', input='hidden0')
# graph.add_node(Dense(50, activation='sigmoid'), name='hidden2', input='hidden1')
graph.add_node(Dense(1, activation='sigmoid', W_regularizer=l2(r)), name='output', input='hidden0', create_output=True)

graph.compile(optimizer='adagrad', loss={'output': 'mse'})
select_best = SelectBestValidation()
# graph.fit(
#     {'input_features': input_features, 'output': (output - 1) / 2},
#     batch_size=256, nb_epoch=50, verbose=1, validation_split=0.5, shuffle=True)
graph.fit(
    {'product_uid': product_uid, 'query_vector': train_query_vectors,
     'output': (output - 1) / 2},
    batch_size=5000, nb_epoch=500, verbose=1, shuffle=True, callbacks=[select_best], validation_split=0.5)

quit()


# TODO Rewrite with new structure
def predict_on_test(df_train, df_test):
    # Setting offset for product_id input
    len_train = df_train.shape[0]
    product_id = {}
    k = prod.keys()
    for id, prod_id in enumerate(k):
        product_id[prod_id] = id + 1

    test_product_uid = df_test['product_uid'].map(lambda x: product_id[x]).values.reshape(len(df_test), 1)
    test_query_vectors = query_vectors[len_train:]
    test_output = df_test['relevance'].values

    y_pred = graph.predict_on_batch(
        {'product_uid': test_product_uid, 'query_vector': test_query_vectors})[
        'output']

    y_pred = y_pred * 2 + 1

    y_pred = y_pred.reshape(len(df_test))

    pd.DataFrame({"id": df_test['id'].values, "relevance": y_pred}).to_csv('submission_nn_03_new.csv', index=False)

    tuned_weights = graph.get_weights()[0]
    # tuned_weights = tuned_weights

    train_X = np.concatenate((tuned_weights[product_uid.reshape((len(product_uid)))]), axis=1)
    train_X = np.concatenate((train_X, train_query_vectors), axis=1)

    train_Y = output

    test_X = np.concatenate((tuned_weights[test_product_uid.reshape((len(test_product_uid)))]), axis=1)
    test_X = np.concatenate((test_X, test_query_vectors), axis=1)

    t_X = pd.DataFrame(train_X)
    t_X.columns = ['feature_' + str(i) for i in range(114)]
    t_X['relevance'] = output

    t_X.to_csv(OUTPUT_PATH + 'X_features.csv', index=False)

    t_Y = pd.DataFrame(test_X)
    t_Y.columns = ['feature_' + str(i) for i in range(114)]
    t_Y.to_csv(OUTPUT_PATH + 'X_features_test.csv', index=False)


# Evaluating error entries
def predict_evaluate_error_terms(df_train):
    train_y_pred = \
        graph.predict({'product_uid': product_uid, 'query_vector': train_query_vectors})[
            'output']
    train_y_pred = train_y_pred * 2 + 1
    train_y_pred = train_y_pred.reshape((len(train_y_pred)))
    train_error = output - train_y_pred

    high_error_indexes = np.where(np.abs(train_error) > 1)[0]

    df_err = df_train.iloc[high_error_indexes]
    df_err['predicted'] = train_y_pred[high_error_indexes]

    df_err.to_csv(OUTPUT_PATH + 'high_error.csv', index=False)
