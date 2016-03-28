from gensim.models.doc2vec import TaggedDocument
from sklearn import pipeline
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
from sklearn.decomposition import TruncatedSVD, NMF, PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.pipeline import FeatureUnion
import logging
import Levenshtein
from utilities import cust_txt_col, cust_regression_vals

logging.getLogger().setLevel(logging.INFO)

__author__ = 'mudit'

from load_preprocessed import *
import numpy as np
from gensim.models import Doc2Vec
from nltk.util import ngrams
from generate_dataset import save_feature

print('- Data and Modules Loaded')

edit_ratio = lambda x, y: Levenshtein.ratio(x, y)
edit_seqratio = lambda x, y: Levenshtein.seqratio(x, y)


def min_edit_dist(query, text):
    t_query = query.split()
    t_text = text.split()
    s = 0
    for t in t_query:
        s += min([Levenshtein.distance(t, token) for token in t_text])
    return s


def str_common_word(str1, str2):
    words, cnt = str1.split(), 0
    for word in words:
        if str2.find(word) >= 0:
            cnt += 1
    return cnt


def str_whole_word(str1, str2, i_):
    cnt = 0
    while i_ < len(str2):
        i_ = str2.find(str1, i_)
        if i_ == -1:
            return cnt
        else:
            cnt += 1
            i_ += len(str1)
    return cnt


def ngram_match(query, string):
    q_tokens = query.split(' ')
    max_n = len(q_tokens)
    total = 0
    similar = 0
    for i in range(1, max_n + 1):
        ngms = ngrams(q_tokens, i)
        for gram in ngms:
            if string.find(" ".join(gram)) >= 0:
                similar += 1
            total += 1
    return similar / float(total)


df_all = df_all.fillna(' ')

# Extracting common features

# Length Features
for f in raw_features:
    df_all['len_of_%s' % f] = df_all[f].map(lambda x: len(x.split())).astype(np.int64)
    save_feature(df_all['len_of_%s' % f], 'len_of_%s' % f)

df_all['product_info'] = df_all['search_term'] + "\t" + df_all['product_title'] + "\t" + df_all['product_description']
df_all['attr'] = df_all['search_term'] + "\t" + df_all['brand']

# df_all['query_in_title'] = df_all['product_info'].map(lambda x: str_whole_word(x.split('\t')[0], x.split('\t')[1], 0))
# df_all['query_in_description'] = df_all['product_info'].map(
#     lambda x: str_whole_word(x.split('\t')[0], x.split('\t')[2], 0))

save_feature(df_all.apply(lambda x: str_whole_word(x[key_query], x[key_title], 0), 1), 'query_in_title')
save_feature(df_all.apply(lambda x: str_whole_word(x[key_query], x[key_description], 0), 1), 'query_in_description')

# Edit Distance and Ratio
for f in [key_description, key_title]:
    # Edit Distance
    df_all['edit_dist_in_%s' % f] = df_all.apply(
        lambda x: min_edit_dist(x[key_query], x[f]), 1)
    save_feature(df_all['edit_dist_in_%s' % f], 'edit_dist_in_%s' % f)
    # Edit Ratio
    df_all['edit_ratio_in_%s' % f] = df_all['edit_dist_in_%s' % f] / df_all['len_of_%s' % f]
    save_feature(df_all['edit_ratio_in_%s' % f], 'edit_ratio_in_%s' % f)

# TODO Change this to new format
df_all['edit_dist_in_info'] = df_all['product_info'].map(
    lambda x: min_edit_dist(x.split('\t')[0], x.split('\t')[1] + ' ' + x.split('\t')[2]))
df_all['edit_ratio_in_info'] = df_all['edit_dist_in_info'] / df_all['len_of_%s' % key_query]

save_feature(df_all['edit_dist_in_info'], 'edit_dist_in_info')
save_feature(df_all['edit_ratio_in_info'], 'edit_ratio_in_info')

# Common Words, and Edit Similarity
for f in [key_title, key_description, key_brand]:
    df_all['word_in_%s' % f] = df_all.apply(lambda x: str_common_word(x[key_query], x[f]), 1)
    df_all['edit_in_%s' % f] = df_all.apply(lambda x: edit_ratio(x[key_query], x[f]), 1)
    df_all['seq_edit_in_%s' % f] = df_all.apply(lambda x: edit_seqratio(x[key_query], x[f]), 1)
    # Saving Features
    save_feature(df_all['word_in_%s' % f], 'word_in_%s' % f)
    save_feature(df_all['edit_in_%s' % f], 'edit_in_%s' % f)
    save_feature(df_all['seq_edit_in_%s' % f], 'seq_edit_in_%s' % f)

for f in [key_title, key_description, key_brand]:
    save_feature(df_all['word_in_%s' % f] / df_all['len_of_%s' % key_query], 'word_ratio_in_%s' % f)

for f in [key_title, key_description, key_brand]:
    df_all['ngram_match_%s' % f] = df_all.apply(lambda x: ngram_match(x[key_query], x[f]), 1)
    save_feature(df_all['ngram_match_%s' % f], 'ngram_match_%s' % f)

df_brand = pd.unique(df_all.brand.ravel())
d = {}
i = 1
for s in df_brand:
    d[s] = i
    i += 1

df_all['brand_feature'] = df_all['brand'].map(lambda x: d[x])
df_all['search_term_feature'] = df_all['search_term'].map(lambda x: len(x))

save_feature(df_all['brand_feature'], '%s_features' % key_brand)
save_feature(df_all['search_term_feature'], '%s_features' % key_query)

print('- Common Features Extracted')

df_train = df_all[:len_train]
df_test = df_all[len_train:]

df_train.to_csv(INPUT_PATH + "df_train2.csv", index=False)
df_test.to_csv(INPUT_PATH + "df_test2.csv", index=False)

# Generating TF-IDF Vectors
tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
tsvd = TruncatedSVD(n_components=50, random_state=2016)

# Training with whole the corpus at once
tfidf.fit(cust_txt_col(key='product_info').transform(df_all))

sparse = tfidf.transform(cust_txt_col(key=key_query).transform(df_all))
query_vector_tfidf_50 = tsvd.fit_transform(sparse)
save_feature(query_vector_tfidf_50, 'tfidf_svd50_%s_vector' % key_query)

sparse = tfidf.transform(cust_txt_col(key=key_title).transform(df_all))
title_vector_tfidf_50 = tsvd.fit_transform(sparse)
save_feature(title_vector_tfidf_50, 'tfidf_svd50_%s_vector' % key_title)

sparse = tfidf.transform(cust_txt_col(key=key_description).transform(df_all))
description_vector_tfidf_50 = tsvd.fit_transform(sparse)
save_feature(description_vector_tfidf_50, 'tfidf_svd50_%s_vector' % key_description)

tsvd = TruncatedSVD(n_components=50, random_state=2016)
pipe2 = pipeline.Pipeline([('s2', cust_txt_col(key=key_title)), ('tfidf2', tfidf), ('tsvd2', tsvd)])
title_vector_tfidf_50 = pipe2.fit_transform(df_all)
save_feature(title_vector_tfidf_50, 'tfidf_svd50_%s_vector' % key_title)

id_test = df_test['id']
y_train = df_train['relevance'].values
X_train = df_train[:].fillna(0)
X_test = df_test[:].fillna(0)

# X_train.dump(DATASET_PATH + 'X_train2/.numpy')
# X_test.dump(INPUT_PATH + 'X_test2.numpy')
y_train.dump(DATASET_PATH + 'Y_train.npy')
id_test.values.dump(DATASET_PATH + 'id_test.npy')

# Implementing Doc2Vec::Gensim
# print("- Extracting Doc2Vec Features")

# def array_to_document(sources):
#     sentences = []
#     for id, source in enumerate(sources):
#         sentences.append(TaggedDocument(source.split(), ['doc_' + str(id)]))
#     return sentences

# print('\t- Preparing data')

# product_info = (df_all['search_term'] + " " + df_all['product_title'] + " " + df_all['product_description']).astype(str)
# product_info = array_to_document(product_info)
#
# model = Doc2Vec(size=100, window=8, min_count=5, workers=384, alpha=0.025, min_alpha=0.025)
# model.build_vocab(product_info)
#
# for epoch in range(10):
#     print("\t- Train #" + str(epoch + 1))
#     model.train(product_info)
#     model.alpha -= 0.0002  # decrease the learning rate
#     model.min_alpha = model.alpha  # fix the learning rate, no decay
#
# print('\t- Saving Doc2Vec model')
# model.save(INPUT_PATH + 'doc2vec')

# model = Doc2Vec.load(INPUT_PATH + 'doc2vec')

# print('\t- Loaded Doc2Vec model')

# weights = [model.docvecs['doc_' + str(id)] for id in range(len(product_info))]
# weights = pd.DataFrame(weights)
# weights.columns = weights.columns.to_series().map(lambda x: 'feature_' + str(x))

# df_all = pd.concat([df_all, weights], axis=1)
