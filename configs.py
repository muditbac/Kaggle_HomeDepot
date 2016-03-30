INPUT_PATH = "../input/"

OUTPUT_PATH = "../output/"

FEATURES_PATH = "features/"

DATASET_PATH = 'datasets/'

configs = {
    'n_folds': 4,
    'seed': 2016,
    'silent': False
}

LEN_TRAIN = 74067

TRAIN_FILE = INPUT_PATH + 'train.csv'
TEST_FILE = INPUT_PATH + 'test.csv'

# Raw Feature Names
key_query = 'search_term'
key_title = 'product_title'
key_description = 'product_description'
key_brand = 'brand'
raw_features = [key_query, key_title, key_description, key_brand]

DEBUG = True

if DEBUG:
    PROCESS_PATH = 'processing/'
    FOLD_PATH = PROCESS_PATH + str(configs['n_folds'])+'folds/'
    nFoldFilename = str(configs['n_folds']) + 'fold_config.pickle'
    TEST_CUTOFF = 0.20
    mConfig = {
        'rfr_n_trees': 100,
        'etr_n_trees': 100,
        'gbr_n_trees': 100,
        'xgb_n_trees:linear': 263,
        'xgb_b_trees:logistic': 263
    }
    NJOBS = -1
else:
    PROCESS_PATH = 'processing/'
    FOLD_PATH = PROCESS_PATH + str(configs['n_folds'])+'folds/'
    nFoldFilename = str(configs['n_folds']) + 'fold_config.pickle'
    TEST_CUTOFF = 0.20 # Calibrated with leaderboard +- 0.001
    mConfig = {
        'rfr_n_trees': 100,
        'etr_n_trees': 100,
        'gbr_n_trees': 100,
        'xgb_n_trees:linear': 263,
        'xgb_b_trees:logistic': 263
    }
    NJOBS = -1

float_formatter = lambda x: "%.3f" % x

