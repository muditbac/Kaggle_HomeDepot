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

logging.getLogger().setLevel(logging.INFO)

__author__ = 'mudit'

from load_preprocessed import *
import numpy as np
from gensim.models import Doc2Vec
from nltk.util import ngrams
print('- Data and Modules Loaded')

edit_ratio = lambda x, y: Levenshtein.ratio(x, y)
edit_seqratio = lambda x, y: Levenshtein.seqratio(x, y)
edit_setratio = lambda x, y: Levenshtein.setratio(x, y)


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
    for i in range(1,max_n+1):
        ngms = ngrams(q_tokens, i)
        for gram in ngms:
            if string.find(" ".join(gram))>=0:
                similar += 1
            total += 1
    return similar / float(total)

df_all = df_all.fillna('')

# Extracting common features
df_all['len_of_query'] = df_all['search_term'].map(lambda x: len(x.split())).astype(np.int64)
df_all['len_of_title'] = df_all['product_title'].map(lambda x: len(x.split())).astype(np.int64)
df_all['len_of_description'] = df_all['product_description'].map(lambda x: len(x.split())).astype(np.int64)
df_all['len_of_brand'] = df_all['brand'].map(lambda x: len(x.split())).astype(np.int64)

df_all['product_info'] = df_all['search_term'] + "\t" + df_all['product_title'] + "\t" + df_all['product_description']
df_all['attr'] = df_all['search_term'] + "\t" + df_all['brand']

df_all['query_in_title'] = df_all['product_info'].map(lambda x: str_whole_word(x.split('\t')[0], x.split('\t')[1], 0))
df_all['query_in_description'] = df_all['product_info'].map(
    lambda x: str_whole_word(x.split('\t')[0], x.split('\t')[2], 0))
df_all['edit_ratio_in_title'] = df_all['product_info'].map(
    lambda x: min_edit_dist(x.split('\t')[0], x.split('\t')[1] + ' ' + x.split('\t')[2]))

df_all['word_in_title'] = df_all['product_info'].map(lambda x: str_common_word(x.split('\t')[0], x.split('\t')[1]))
df_all['edit_in_title'] = df_all['product_info'].map(lambda x: edit_ratio(x.split('\t')[0], x.split('\t')[1]))
df_all['seq_edit_in_title'] = df_all['product_info'].map(lambda x: edit_seqratio(x.split('\t')[0], x.split('\t')[1]))
df_all['word_in_description'] = df_all['product_info'].map(
    lambda x: str_common_word(x.split('\t')[0], x.split('\t')[2]))
df_all['word_in_brand'] = df_all['attr'].map(lambda x: str_common_word(x.split('\t')[0], x.split('\t')[1]))

df_all['ratio_title'] = df_all['word_in_title'] / df_all['len_of_query']
df_all['ratio_description'] = df_all['word_in_description'] / df_all['len_of_query']
df_all['ratio_brand'] = df_all['word_in_brand'] / df_all['len_of_brand']

df_brand = pd.unique(df_all.brand.ravel())
d = {}
i = 1
for s in df_brand:
    d[s] = i
    i += 1
df_all['brand_feature'] = df_all['brand'].map(lambda x: d[x])
df_all['search_term_feature'] = df_all['search_term'].map(lambda x: len(x))

print('- Common Features Extracted')

df_all = df_all.fillna('')
df_train = df_all[:len_train]
df_test = df_all[len_train:]

df_train.to_csv(INPUT_PATH + "df_train2.csv", index=False)
df_test.to_csv(INPUT_PATH + "df_test2.csv", index=False)



# Implementing Doc2Vec::Gensim
print("- Extracting Doc2Vec Features")


def array_to_document(sources):
    sentences = []
    for id, source in enumerate(sources):
        sentences.append(TaggedDocument(source.split(), ['doc_' + str(id)]))
    return sentences


print('\t- Preparing data')

product_info = (df_all['search_term'] + " " + df_all['product_title'] + " " + df_all['product_description']).astype(str)
product_info = array_to_document(product_info)
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

model = Doc2Vec.load(INPUT_PATH + 'doc2vec')

print('\t- Loaded Doc2Vec model')

weights = [model.docvecs['doc_' + str(id)] for id in range(len(product_info))]
weights = pd.DataFrame(weights)
weights.columns = weights.columns.to_series().map(lambda x: 'feature_' + str(x))

df_all = pd.concat([df_all, weights], axis=1)


# -----------------------------------------------------------------------------------
# Random Forest Regressor
def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions) ** 0.5
    return fmean_squared_error_


RMSE = make_scorer(fmean_squared_error, greater_is_better=False)

feature_cols = ['feature_' + str(x) for x in range(100)]


class cust_regression_vals(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, hd_searches):
        d_col_drops = ['id', 'relevance', 'search_term', 'product_title', 'product_description', 'brand',
                       'search_and_prod_info'] + feature_cols
        hd_searches = hd_searches.drop(d_col_drops, axis=1).values
        return hd_searches


class cust_txt_col(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


df_all['search_and_prod_info'] = (df_all['search_term'] + " " + df_all['product_title'] + " " + df_all[
    'product_description'])  # .map(iso_encode).astype(str)

df_train = df_all[:len_train]
df_test = df_all[len_train:]

rfr = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=2016, verbose=1)
tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', encoding='ISO-8859-1')
tsvd = TruncatedSVD(n_components=10, random_state=2016)
tnmf = NMF(n_components=10, random_state=2016)
tpca = PCA(n_components=10)

clf = pipeline.Pipeline([
    ('union', FeatureUnion(
        transformer_list=[
            # ('cst',  cust_regression_vals()),
            ('txt1', pipeline.Pipeline([('s1', cust_txt_col(key='search_term')), ('tfidf1', tfidf), ('tsvd1', tsvd)])),
            ('txt2',
             pipeline.Pipeline([('s2', cust_txt_col(key='search_and_prod_info')), ('tfidf2', tfidf), ('tsvd2', tsvd)])),
            ('txt3',
             pipeline.Pipeline([('s3', cust_txt_col(key='search_and_prod_info')), ('tfidf3', tfidf), ('tnmf', tnmf)])),
            ('txt4', pipeline.Pipeline([('s4', cust_txt_col(key='brand')), ('tfidf4', tfidf), ('tsvd4', tsvd)]))
        ],
        transformer_weights={
            'cst': 1.0,
            'txt1': 0.5,
            'txt2': 0.75,
            'txt3': 0.75,
            'txt4': 0.5
        },
        # n_jobs = -1
    )),
    ('rfr', rfr)])

param_grid = {'rfr__max_features': [25], 'rfr__max_depth': [25]}
model = GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=-1, cv=2, verbose=20, scoring=RMSE)
model.fit(df_train, df_train['relevance'].values)

y_pred = model.predict(df_test)


for i in range(len(y_pred)):
    if y_pred[i]<1:
        y_pred[i] = 1
    elif y_pred[i]>3:
        y_pred[i] = 3


pd.DataFrame({"id": df_test['id'].values, "relevance": y_pred}).to_csv('submission_rfr_spell_02.csv', index=False)
