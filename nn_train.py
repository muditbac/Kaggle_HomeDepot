from configs import *
import numpy as np
np.random.seed(configs['seed'])

import pickle
import logging

import pandas as pd
import theano

from generate_dataset import save_feature

from keras.layers import Embedding, Flatten, Dense, LSTM
from keras.models import Graph
from keras.regularizers import l2
from keras.callbacks import Callback

__author__ = 'mudit'

logging.getLogger().setLevel(logging.INFO)


class SelectBestValidation(Callback):
    def __init__(self, graph):
        super(SelectBestValidation, self).__init__()
        self.graph = graph

    def on_train_begin(self, logs={}):
        self.best_validation_score = -1

    def on_epoch_end(self, epoch, logs={}):
        val_score = logs.get('val_loss')
        if self.best_validation_score == -1 or val_score <= self.best_validation_score:
            self.best_validation_score = val_score
            self.weights = self.graph.get_weights()
            print('Saving model for val_score = %f' % val_score)

def train_keras_nn():
    r = 1e-3
    graph = Graph()
    # graph.add_input(name='input_features', input_shape=(input_features.shape[1],))
    graph.add_input(name='train_product_nnid', input_shape=(1,), dtype=int)
    graph.add_input(name='query_indexes', input_shape=(15,), dtype=int)
    graph.add_node(Embedding(input_dim=len(prod) + 1,
                             output_dim=100,
                             weights=[np.concatenate((np.zeros((1, 100)), product_vectors), axis=0)],
                             trainable=True,
                             input_length=1,
                             W_regularizer=l2(r)),
                   name='embedding',
                   input='train_product_nnid', )
    graph.add_node(Flatten(), name='flatten', input='embedding')

    graph.add_node(Embedding(input_dim=len(vocab_vectors) + 1,
                             output_dim=50,
                             weights=[np.concatenate((np.zeros((1, 50)), vocab_vectors), axis=0)],
                             trainable=True,
                             input_length=1,
                             W_regularizer=l2(r), mask_zero=True),
                   name='q_embedding',
                   input='query_indexes', )
    lstm = LSTM(output_dim=50)
    graph.add_node(lstm, name='LSTM', input='q_embedding')
    graph.add_node(Dense(100, activation='sigmoid', W_regularizer=l2(r)), name='hidden0',
                   inputs=['LSTM', 'flatten'],
                   merge_mode="concat", concat_axis=1)
    graph.add_node(Dense(1, activation='sigmoid', W_regularizer=l2(r)), name='output', input='hidden0', create_output=True)

    graph.compile(optimizer='adam', loss={'output': 'mse'})

    select_best = SelectBestValidation(graph)

    # To get weights of query vectors
    get_lstm_output = theano.function([graph.inputs['query_indexes'].input], lstm.get_output(train=False))

    graph.fit(
        {'train_product_nnid': train_product_nnid, 'query_indexes': query_train,
         'output': (output - 1) / 2},
        batch_size=1000, nb_epoch=100, verbose=1, shuffle=True, callbacks=[select_best], validation_split=0.1)

    return graph, get_lstm_output

# TODO Rewrite with new structure
def predict_on_test(df_train, df_test, graph, ):
    # Setting offset for product_id input
    # len_train = df_train.shape[0]
    # product_id = {}
    # k = prod.keys()
    # for id, prod_id in enumerate(k):
    #     product_id[prod_id] = id + 1
    #
    # test_product_uid = df_test['train_product_nnid'].map(lambda x: product_id[x]).values.reshape(len(df_test), 1)
    # test_query_vectors = query_vectors[len_train:]
    # test_output = df_test['relevance'].values
    #
    # y_pred = graph.predict_on_batch(
    #     {'train_product_nnid': test_product_uid, 'query_vector': test_query_vectors})[
    #     'output']
    #
    # y_pred = y_pred * 2 + 1
    #
    # y_pred = y_pred.reshape(len(df_test))
    #
    # pd.DataFrame({"id": df_test['id'].values, "relevance": y_pred}).to_csv('submission_nn_03_new.csv', index=False)
    #
    # tuned_weights = graph.get_weights()[0]
    # # tuned_weights = tuned_weights
    #
    # train_X = np.concatenate((tuned_weights[train_product_nnid.reshape((len(train_product_nnid)))]), axis=1)
    # train_X = np.concatenate((train_X, train_query_vectors), axis=1)
    #
    # train_Y = output
    #
    # test_X = np.concatenate((tuned_weights[test_product_uid.reshape((len(test_product_uid)))]), axis=1)
    # test_X = np.concatenate((test_X, test_query_vectors), axis=1)
    #
    # t_X = pd.DataFrame(train_X)
    # t_X.columns = ['feature_' + str(i) for i in range(114)]
    # t_X['relevance'] = output
    #
    # t_X.to_csv(OUTPUT_PATH + 'X_features.csv', index=False)
    #
    # t_Y = pd.DataFrame(test_X)
    # t_Y.columns = ['feature_' + str(i) for i in range(114)]
    # t_Y.to_csv(OUTPUT_PATH + 'X_features_test.csv', index=False)
    pass

# TODO Rewrite with new structure
# Evaluating error entries
def predict_evaluate_error_terms(df_train, graph):
    train_y_pred = \
        graph.predict({'train_product_nnid': train_product_nnid, 'query_vector': query_train})[
            'output']
    train_y_pred = train_y_pred * 2 + 1
    train_y_pred = train_y_pred.reshape((len(train_y_pred)))
    train_error = output - train_y_pred

    high_error_indexes = np.where(np.abs(train_error) > 1)[0]

    df_err = df_train.iloc[high_error_indexes]
    df_err['predicted'] = train_y_pred[high_error_indexes]

    df_err.to_csv(OUTPUT_PATH + 'high_error.csv', index=False)


def save_tuned_weights(graph, get_lstm_output):
    all_weights = graph.get_weights()
    product_weights = all_weights[0]
    df_all = np.concatenate([train_product_nnid, test_product_nnid])

    # Reshape df_all
    df_all.shape = len(df_all)
    prod_vecs = product_weights[df_all]
    save_feature(prod_vecs, 'dnn_product_vector')

    query_all = np.concatenate([query_train, query_test])

    # Dividing prediction into buckets (GPU out of memory)
    query_vecs = []
    len_all = len(query_all)
    size = 10000
    n = len_all/size + 1
    for i in range(n):
        l = get_lstm_output(query_all[size*i:size*(i+1)])
        query_vecs.append(l)
    query_vecs = np.concatenate(query_vecs)

    # Saving Query Vectors
    save_feature(query_vecs, 'dnn_query_vector')


if __name__ == "__main__":
    df_train = pd.read_csv(INPUT_PATH + 'df_train2.csv')
    df_test = pd.read_csv(INPUT_PATH + 'df_test2.csv')
    [train_product_nnid, test_product_nnid, output, product_vectors, prod, vocab_vectors, query_train, query_test] = pickle.load(
        open(OUTPUT_PATH + 'nn.pickle', 'rb'))


    model, get_lstm_output = train_keras_nn()
    print('- Deep Net Trained')
    save_tuned_weights(model, get_lstm_output)
    print('- Tuned Weights Saved')
