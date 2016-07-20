from gensim.models import Word2Vec
from configs import INPUT_PATH
from load_preprocessed import *
import numpy as np
from generate_dataset import save_feature
from glove import  Glove

__author__ = 'mudit'



model = Word2Vec.load_word2vec_format(INPUT_PATH + 'GoogleNews-vectors-negative300.bin.gz',
                                      binary=True)  # C binary format


def calculate_n_similarity(x, key1, key2):
    w1 = [w for w in x[key1].split() if w in model.vocab]
    w2 = [w for w in x[key2].split() if w in model.vocab]
    res = model.n_similarity(w1, w2)
    if isinstance(res, np.ndarray):
        return 0
    else:
        return float(res)


for key in [key_title, key_description]:
    df_all['word2vec_n_similarity_%s_%s' % (key_query, key)] = df_all.apply(
        lambda x: calculate_n_similarity(x, key_query, key), 1)
    save_feature(df_all['word2vec_n_similarity_%s_%s' % (key_query, key)], 'word2vec_n_similarity_%s_%s' % (key_query, key))

# model = Glove.load_stanford(INPUT_PATH + 'glove.6B.50d.txt')
#
#
# for key in [key_title, key_description]:
#     df_all['word2vec_n_similarity_%s_%s' % (key_query, key)] = df_all.apply(
#         lambda x: calculate_n_similarity(x, key_query, key), 1)
#     save_feature(df_all['word2vec_n_similarity_%s_%s' % (key_query, key)], 'word2vec_n_similarity_%s_%s' % (key_query, key))
