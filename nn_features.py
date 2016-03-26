import pickle
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

__author__ = 'mudit'

import pandas as pd
import keras
from keras.layers import Dense, MaxoutDense, Embedding, Flatten
from keras.models import Graph
from keras.objectives import *
import logging
from configs import *
import numpy as np
from keras.optimizers import *
import pickle
logging.getLogger().setLevel(logging.INFO)

df_train = pd.read_csv(INPUT_PATH + 'df_train2.csv')
df_test = pd.read_csv(INPUT_PATH + 'df_test2.csv')

len_train = len(df_train)

df_all = pd.concat([df_train, df_test]).fillna('')

df_products = df_all[['product_uid', 'product_title', 'product_description', 'brand']]

# Finding unique products
prod = {}
for row in df_products.iterrows():
    prod[row[1]['product_uid']] = row[1]['product_title'] + ' ' + row[1]['product_description'] + ' ' + row[1]['brand']

print('Creating features...')

tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
tsvd = TruncatedSVD(n_components=50, random_state=2016)

tfidf.fit((df_all['product_info'] + '\t' + df_all['brand']).values)
print('Training tfidf...')

product_vectors = tfidf.transform(prod.values())
product_vectors = tsvd.fit_transform(product_vectors)
print('Generating product_vectors...')

query_vectors = tfidf.transform(df_all['search_term'])
query_vectors = tsvd.fit_transform(query_vectors)
print('Generating query_vectors')

# Setting offset for product_id input
product_id = {}
k = prod.keys()
for id, prod_id in enumerate(k):
    product_id[prod_id] = id + 1






# Neural Network
product_uid = df_train['product_uid'].map(lambda x: product_id[x]).values.reshape(len_train, 1)
input_features = df_train[['len_of_query', 'len_of_title',
                           'len_of_description', 'len_of_brand',
                           'query_in_title', 'query_in_description', 'word_in_title',
                           'word_in_description', 'word_in_brand', 'ratio_title',
                           'ratio_description', 'ratio_brand', 'brand_feature',
                           'search_term_feature']].values
output = df_train['relevance'].values
train_query_vectors = query_vectors[:len_train]

pickle.dump([product_uid, input_features, output, train_query_vectors, query_vectors, prod, product_vectors], open(OUTPUT_PATH + 'nn.pickle','wb'))

graph = Graph()
graph.add_input(name='input_features', input_shape=(14,))
graph.add_input(name='product_uid', input_shape=(1,), dtype=int)
graph.add_input(name='query_vector', input_shape=(50,))
graph.add_node(Embedding(input_dim=len(prod),
                         output_dim=50,
                         weights=[product_vectors],
                         trainable=True,
                         input_length=1),
               name='embedding',
               input='product_uid', )
graph.add_node(Flatten(), name='flatten', input='embedding')
graph.add_node(Dense(50, activation='tanh'), name='hidden', inputs=['input_features', 'query_vector', 'flatten'],
               merge_mode="concat", concat_axis=1)
graph.add_node(Dense(1), name='output', input='hidden', create_output=True)

graph.compile(optimizer=Adam(epsilon=10e-4), loss={'output': 'mse'})

graph.fit(
    {'input_features': input_features, 'product_uid': product_uid, 'query_vector': train_query_vectors,
     'output': output},
    batch_size=256, nb_epoch=200, verbose=1, validation_split=0.1)

[product_uid, input_features, output, train_query_vectors, query_vectors, prod, product_vectors] = pickle.load(open(INPUT_PATH+'nn.pickle'))