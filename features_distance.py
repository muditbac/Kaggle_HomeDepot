import ngram

__author__ = 'mudit'

from load_preprocessed import *
import numpy as np
from configs import *

def try_divide(x, y, val=0.0):
    """
    	Try to divide two numbers
    """
    if y != 0.0:
        val = float(x) / y
    return val


#####################
## Distance metric ##
#####################
def JaccardCoef(A, B):
    A, B = set(A), set(B)
    intersect = len(A.intersection(B))
    union = len(A.union(B))
    coef = try_divide(intersect, union)
    return coef


def DiceDist(A, B):
    A, B = set(A), set(B)
    intersect = len(A.intersection(B))
    union = len(A) + len(B)
    d = try_divide(2 * intersect, union)
    return d


def compute_dist(A, B, dist="jaccard_coef"):
    if dist == "jaccard_coef":
        d = JaccardCoef(A, B)
    elif dist == "dice_dist":
        d = DiceDist(A, B)
    return d


#### pairwise distance
def pairwise_jaccard_coef(A, B):
    coef = np.zeros((A.shape[0], B.shape[0]), dtype=float)
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            coef[i, j] = JaccardCoef(A[i], B[j])
    return coef


def pairwise_dice_dist(A, B):
    d = np.zeros((A.shape[0], B.shape[0]), dtype=float)
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            d[i, j] = DiceDist(A[i], B[j])
    return d


def pairwise_dist(A, B, dist="jaccard_coef"):
    if dist == "jaccard_coef":
        d = pairwise_jaccard_coef(A, B)
    elif dist == "dice_dist":
        d = pairwise_dice_dist(A, B)
    return d

#####################################
## Extract basic distance features ##
#####################################
def extract_basic_distance_feat(df):
    ## unigram
    print("generate unigram")

    # for gram in ['unigram', 'bigram', 'trigram']:
    for f in [key_query, key_title, key_description]:
        df["%s_%s" % ('unigram', f)] = list(df.apply(lambda x: x[f].split(), axis=1))

    ## bigram
    print("generate bigram")
    join_str = "_"
    for f in [key_query, key_title, key_description]:
        df["%s_%s" % ('bigram', f)] = list(df.apply(lambda x: ngram.getBigram(x['%s_%s' % ('unigram', f)], join_str), axis=1))

    # df["query_bigram"] = list(df.apply(lambda x: ngram.getBigram(x["query_unigram"], join_str), axis=1))
    # df["title_bigram"] = list(df.apply(lambda x: ngram.getBigram(x["title_unigram"], join_str), axis=1))
    # df["description_bigram"] = list(df.apply(lambda x: ngram.getBigram(x["description_unigram"], join_str), axis=1))
    ## trigram
    print("generate trigram")
    join_str = "_"
    for f in [key_query, key_title, key_description]:
        df["%s_%s" % ('trigram', f)] = list(df.apply(lambda x: ngram.getTrigram(x['%s_%s' % ('unigram', f)], join_str), axis=1))
    # df["query_trigram"] = list(df.apply(lambda x: ngram.getTrigram(x["query_unigram"], join_str), axis=1))
    # df["title_trigram"] = list(df.apply(lambda x: ngram.getTrigram(x["title_unigram"], join_str), axis=1))
    # df["description_trigram"] = list(df.apply(lambda x: ngram.getTrigram(x["description_unigram"], join_str), axis=1))

    ## jaccard coef/dice dist of n-gram
    print("generate jaccard coef and dice dist for n-gram")
    dists = ["jaccard_coef", "dice_dist"]
    grams = ["unigram", "bigram", "trigram"]
    feat_names = [key_query, key_title, key_description]
    for dist in dists:
        for gram in grams:
            for i in range(len(feat_names)-1):
                for j in range(i+1,len(feat_names)):
                    target_name = feat_names[i]
                    obs_name = feat_names[j]
                    df["%s_of_%s_between_%s_%s"%(dist,gram,target_name,obs_name)] = \
                            list(df.apply(lambda x: compute_dist(x[gram+"_"+target_name], x[gram+"_"+obs_name], dist), axis=1))


extract_basic_distance_feat(df_all)