__author__ = 'mudit'

from local_paths import *

import pandas as pd

df_train = pd.read_csv(INPUT_PATH + "df_train.csv", encoding="ISO-8859-1", index_col = 0)
df_test = pd.read_csv(INPUT_PATH + "df_test.csv", encoding="ISO-8859-1", index_col = 0)
# df_attributes = pd.read_csv(INPUT_PATH + "df_attr.csv", encoding="ISO-8859-1", index_col = 0)

len_train = len(df_train)

df_all = pd.concat([df_train,df_test])