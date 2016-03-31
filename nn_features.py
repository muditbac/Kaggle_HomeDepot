import pandas as pd

__author__ = 'mudit'

import logging
import pickle

import numpy as np

from configs import *

logging.getLogger().setLevel(logging.INFO)

df_train = pd.read_csv(INPUT_PATH + 'df_train2.csv', encoding="ISO-8859-1", index_col = 0)
len_train = len(df_train)

product_uid = np.load(FEATURES_PATH + 'product_uid.npy')
title_vector = np.load(FEATURES_PATH + 'tfidf_svd50_product_title_vector.npy')
description_vector = np.load(FEATURES_PATH + 'tfidf_svd50_product_description_vector.npy')
query_vectors = np.load(FEATURES_PATH + 'tfidf_svd50_search_term_vector.npy')

product_vector = np.concatenate([title_vector,description_vector],axis=1)


product_id_to_vector = {}
for id, prod_id in enumerate(product_uid):
    product_id_to_vector[prod_id] = product_vector[id]

# Setting offset for product_id input
product_uid_to_nnid = {}
for id, prod_id in enumerate(product_id_to_vector.keys()):
    # print(id)
    # print(prod_id)
    product_uid_to_nnid[prod_id] = id+1




# Neural Network
product_uid = df_train['product_uid'].map(lambda x: product_uid_to_nnid[x]).values.reshape(len_train, 1)
# input_features = df_train[].values
output = df_train['relevance'].values
train_query_vectors = query_vectors[:len_train]

pickle.dump([product_uid, output, train_query_vectors, query_vectors, np.array(product_id_to_vector.values()), product_uid_to_nnid], open(OUTPUT_PATH + 'nn.pickle','wb'))

# TODO Restructure query vector Important
#
# graph = Graph()
# graph.add_input(name='input_features', input_shape=(14,))
# graph.add_input(name='product_uid', input_shape=(1,), dtype=int)
# graph.add_input(name='query_vector', input_shape=(50,))
# graph.add_node(Embedding(input_dim=len(prod),
#                          output_dim=50,
#                          weights=[product_vectors],
#                          trainable=True,
#                          input_length=1),
#                name='embedding',
#                input='product_uid', )
# graph.add_node(Flatten(), name='flatten', input='embedding')
# graph.add_node(Dense(50, activation='tanh'), name='hidden', inputs=['input_features', 'query_vector', 'flatten'],
#                merge_mode="concat", concat_axis=1)
# graph.add_node(Dense(1), name='output', input='hidden', create_output=True)
#
# graph.compile(optimizer=Adam(epsilon=10e-4), loss={'output': 'mse'})
#
# graph.fit(
#     {'input_features': input_features, 'product_uid': product_uid, 'query_vector': train_query_vectors,
#      'output': output},
#     batch_size=256, nb_epoch=200, verbose=1, validation_split=0.1)
#
# [product_uid, input_features, output, train_query_vectors, query_vectors, prod, product_vectors] = pickle.load(open(INPUT_PATH+'nn.pickle'))