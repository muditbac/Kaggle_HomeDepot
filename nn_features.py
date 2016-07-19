import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

__author__ = 'mudit'

import logging
import pickle

import numpy as np

from configs import *

logging.getLogger().setLevel(logging.INFO)

df_train = pd.read_csv(INPUT_PATH + 'df_train2.csv', encoding="ISO-8859-1", index_col=0)
df_test = pd.read_csv(INPUT_PATH + 'df_test2.csv', encoding="ISO-8859-1", index_col=0)
len_train = len(df_train)
len_test = len(df_test)

product_uid = np.load(FEATURES_PATH + 'product_uid.npy')
title_vector = np.load(FEATURES_PATH + 'tfidf_svd50_product_title_vector.npy')
description_vector = np.load(FEATURES_PATH + 'tfidf_svd50_product_description_vector.npy')


# query_vectors = np.load(FEATURES_PATH + 'tfidf_svd50_search_term_vector.npy')

def prepare_product_vectors(product_uid):
    product_vector = np.concatenate([title_vector, description_vector], axis=1)

    product_id_to_vector = {}
    for id, prod_id in enumerate(product_uid):
        product_id_to_vector[prod_id] = product_vector[id]

    # Setting offset for product_id input
    product_uid_to_nnid = {}
    for id, prod_id in enumerate(product_id_to_vector.keys()):
        # print(id)
        # print(prod_id)
        product_uid_to_nnid[prod_id] = id + 1


    # Neural Network
    train_product_nnid = df_train['product_uid'].map(lambda x: product_uid_to_nnid[x]).values.reshape(len_train, 1)
    test_product_nnid = df_test['product_uid'].map(lambda x: product_uid_to_nnid[x]).values.reshape(len_test, 1)
    # input_features = df_train[].values
    output = df_train['relevance'].values
    return train_product_nnid, test_product_nnid, output, product_id_to_vector, product_uid_to_nnid


def prepare_query_vectors(df_train, df_test):
    search_term = df_train[key_query].values
    search_term = np.concatenate([search_term, df_test[key_query].values])
    count = TfidfVectorizer(tokenizer=lambda x: x.split(' '))

    count.fit(search_term)

    vocab = list(count.vocabulary_.keys())
    sparse = count.transform(vocab)
    tsvd = TruncatedSVD(n_components=50, random_state=configs['seed'])
    vocab_vectors = tsvd.fit_transform(sparse)

    word_to_nnindex = {}
    for i, word in enumerate(vocab):
        word_to_nnindex[word] = i + 1

    # Converting words to indexes
    query_indexes = []
    for i, query in enumerate(search_term):
        words = query.split(' ')
        indexes = []
        for word in words:
            indexes.append(word_to_nnindex[word])
        query_indexes.append(indexes)

    # Add Padding to rest
    max_words = max([len(q) for q in query_indexes])
    for i, query_ind in enumerate(query_indexes):
        l = len(query_ind)
        query_indexes[i].extend([0] * (max_words - l))

    query_train = np.array(query_indexes[:len(df_train)], dtype=np.int32)
    query_test = np.array(query_indexes[len(df_train):], dtype=np.int32)
    return vocab_vectors, query_train, query_test


if __name__ == '__main__':
    train_product_nnid, test_product_nnid, output, product_id_to_vector, product_uid_to_nnid = prepare_product_vectors(
        product_uid)
    print('- Product Vector Prepared')

    vocab_vectors, query_train, query_test = prepare_query_vectors(df_train, df_test)
    print('- Query Vector Prepared')

    pickle.dump([train_product_nnid, test_product_nnid, output, np.array(list(product_id_to_vector.values())),
                 product_uid_to_nnid, vocab_vectors, query_train, query_test], open(OUTPUT_PATH + 'nn.pickle', 'wb'))
    print('- Data dumped to nn.pickle')