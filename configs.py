INPUT_PATH = "../input/"

OUTPUT_PATH = "../output/"

FEATURES_PATH = "features/"

DATASET_PATH = 'datasets/'

configs = {
    'n_folds': 4,
    'seed': 2016
}

TRAIN_FILE = INPUT_PATH + 'train.csv'
TEST_FILE = INPUT_PATH + 'test.csv'


DEBUG = True

if DEBUG:
    PROCESS_PATH = 'debug_processing/'
    FOLD_PATH = PROCESS_PATH + str(configs['n_folds'])+'folds/'
    FOLDS_INFO = FOLD_PATH + 'folds.pkl'
    nFoldFilename = str(configs['n_folds']) + 'fold_config.pickle'
    TEST_CUTOFF = 0.5
    mConfig = {
        'rfr_n_trees': 10,
        'etr_n_trees': 10,
        'gbr_n_trees': 10,
        'xgb_n_trees:linear': 10,
        'xgb_b_trees:logistic': 10
    }
    NJOBS = -1
else:
    PROCESS_PATH = 'processing/'
    FOLD_PATH = PROCESS_PATH + str(configs['n_folds'])+'folds/'
    FOLDS_INFO = FOLD_PATH + 'folds.pkl'
    nFoldFilename = str(configs['n_folds']) + 'fold_config.pickle'
    TEST_CUTOFF = 0.5
    mConfig = {
        'rfr_n_trees': 500,
        'etr_n_trees': 475,
        'gbr_n_trees': 459,
        'xgb_n_trees:linear': 263,
        'xgb_b_trees:logistic': 263
    }
    NJOBS = -1

float_formatter = lambda x: "%.3f" % x
