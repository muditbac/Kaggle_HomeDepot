from utilities import SimpleTransform, identity
from generate_dataset import generate_dataset
import numpy as np
__author__ = 'mudit'

# t_fn = np.sqrt
t_fn = lambda x: np.log(1+x)
# t_fn = identity


if __name__ == '__main__':
    features = [
        ('product_uid', SimpleTransform()),
        ('brand_features', SimpleTransform(transformer=t_fn)),
        ('dice_dist_of_bigram_between_product_title_product_description', SimpleTransform(transformer=t_fn)),
        ('dice_dist_of_bigram_between_search_term_product_description', SimpleTransform(transformer=t_fn)),
        ('dice_dist_of_bigram_between_search_term_product_title', SimpleTransform(transformer=t_fn)),
        ('dice_dist_of_trigram_between_product_title_product_description', SimpleTransform(transformer=t_fn)),
        ('dice_dist_of_trigram_between_search_term_product_description', SimpleTransform(transformer=t_fn)),
        ('dice_dist_of_trigram_between_search_term_product_title', SimpleTransform(transformer=t_fn)),
        ('dice_dist_of_unigram_between_product_title_product_description', SimpleTransform(transformer=t_fn)),
        ('dice_dist_of_unigram_between_search_term_product_description', SimpleTransform(transformer=t_fn)),
        ('dice_dist_of_unigram_between_search_term_product_title', SimpleTransform(transformer=t_fn)),
        ('edit_dist_in_info', SimpleTransform(transformer=t_fn)),
        ('edit_dist_in_product_description', SimpleTransform(transformer=t_fn)),
        ('edit_dist_in_product_title', SimpleTransform(transformer=t_fn)),
        ('edit_in_brand', SimpleTransform(transformer=t_fn)),
        ('edit_in_product_description', SimpleTransform(transformer=t_fn)),
        ('edit_in_product_title', SimpleTransform(transformer=t_fn)),
        ('edit_ratio_in_info', SimpleTransform(transformer=t_fn)),
        ('edit_ratio_in_product_description', SimpleTransform(transformer=t_fn)),
        ('edit_ratio_in_product_title', SimpleTransform(transformer=t_fn)),
        ('jaccard_coef_of_bigram_between_product_title_product_description', SimpleTransform(transformer=t_fn)),
        ('ngram_match_product_description', SimpleTransform(transformer=t_fn)),
        ('ngram_match_product_title', SimpleTransform(transformer=t_fn)),
        ('query_in_description', SimpleTransform(transformer=t_fn)),
        ('query_in_title', SimpleTransform(transformer=t_fn)),
        ('search_term_features', SimpleTransform(transformer=t_fn)),
        ('seq_edit_in_brand', SimpleTransform(transformer=t_fn)),
        ('seq_edit_in_product_description', SimpleTransform(transformer=t_fn)),
        ('seq_edit_in_product_title', SimpleTransform(transformer=t_fn)),
        ('tfidf_svd50_product_description_vector', SimpleTransform(transformer=t_fn)),
        ('tfidf_svd50_product_title_vector', SimpleTransform(transformer=t_fn)),
        ('tfidf_svd50_search_term_vector', SimpleTransform(transformer=t_fn)),
        ('word_in_brand', SimpleTransform(transformer=t_fn)),
        ('word_in_product_description', SimpleTransform(transformer=t_fn)),
        ('word_in_product_title', SimpleTransform(transformer=t_fn)),
        ('word_ratio_in_brand', SimpleTransform(transformer=t_fn)),
        ('word_ratio_in_product_description', SimpleTransform(transformer=t_fn)),
        ('word_ratio_in_product_title', SimpleTransform(transformer=t_fn)),
        ('jaccard_coef_of_bigram_between_search_term_product_title', SimpleTransform(transformer=t_fn)),
        ('jaccard_coef_of_trigram_between_product_title_product_description', SimpleTransform(transformer=t_fn)),
        ('jaccard_coef_of_trigram_between_search_term_product_description', SimpleTransform(transformer=t_fn)),
        ('jaccard_coef_of_trigram_between_search_term_product_title', SimpleTransform(transformer=t_fn)),
        ('jaccard_coef_of_unigram_between_product_title_product_description', SimpleTransform(transformer=t_fn)),
        ('jaccard_coef_of_unigram_between_search_term_product_description', SimpleTransform(transformer=t_fn)),
        ('jaccard_coef_of_unigram_between_search_term_product_title', SimpleTransform(transformer=t_fn)),
        ('len_of_brand', SimpleTransform(transformer=t_fn)),
        ('len_of_product_description', SimpleTransform(transformer=t_fn)),
        ('len_of_product_title', SimpleTransform(transformer=t_fn)),
        ('len_of_search_term', SimpleTransform(transformer=t_fn)),
        ('jaccard_coef_of_bigram_between_search_term_product_description', SimpleTransform(transformer=t_fn)),
        ('ngram_match_brand', SimpleTransform(transformer=t_fn)),
        ('word2vec_n_similarity_search_term_product_description', SimpleTransform(transformer=np.nan_to_num)),
        ('word2vec_n_similarity_search_term_product_title', SimpleTransform(transformer=np.nan_to_num)),
    ]

    generate_dataset(features, 'svd50x3_1plogx_product_uid')
