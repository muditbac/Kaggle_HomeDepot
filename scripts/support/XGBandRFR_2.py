import time
start_time = time.time()

import numpy as np
import pandas as pd
from nltk.metrics import edit_distance
import re

def str_common_word(str1, str2):
    str2 = str2.lower().split(" ")
    if str1 in str2:
        cnt=1
    else:
        cnt=0
    return cnt

def str_common_word2(str1, str2):
    str2 = str(str2).lower()
    if str2.find(str1)>=0:
        cnt=1
    else:
        cnt=0
    return cnt

df_train = pd.read_csv('../input/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('../input/test.csv', encoding="ISO-8859-1")
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = df_all[['product_uid','search_term','product_title']]
df_all.reset_index(inplace=True)

df_prod = pd.read_csv('../input/product_descriptions.csv').fillna(" ")
df_attr = pd.read_csv('../input/attributes.csv').fillna(" ")
print("--- Files Loaded: %s minutes ---" % round(((time.time() - start_time)/60),2))

d_prod_query = {}
for i in range(len(df_all)):
    b_ = str(df_all['product_uid'][i])
    if b_ not in d_prod_query:
        d_prod_query[b_] = [list(set(str(df_all['search_term'][i]).lower().split(" "))), 
                            str(df_all['product_title'][i]).lower(),
                            str(df_prod.loc[df_prod['product_uid'] == df_all['product_uid'][i]]['product_description'].iloc[0]).lower()]
    else:
        d_prod_query[b_][0] = list(set(d_prod_query[b_][0] + list(set(str(df_all['search_term'][i]).lower().split(" ")))))

f = open("dictionary.txt", "w")
f.write(str(d_prod_query))
f.close()

print("--- Product & Search Term Dictionary: %s minutes ---" % round(((time.time() - start_time)/60),2))
#stop_ = list(text.ENGLISH_STOP_WORDS)
stop_ = []
d={}
for i in d_prod_query:
    a = d_prod_query[i][0]
    df_gen_attr = df_attr.loc[df_attr['product_uid'] == int(i)]
    for b_ in a:
        if len(b_)>0:
            col_lst = []
            for j in range(len(df_gen_attr)):
                if str_common_word(b_, df_gen_attr['value'].iloc[j])>0:
                    col_lst.append(df_gen_attr['name'].iloc[j])
            #if b_ not in stop_:
            if b_ not in d:
                d[b_] = [1,str_common_word(b_, d_prod_query[i][1]),str_common_word2(b_, d_prod_query[i][1]),col_lst[:]]
            else:
                d[b_][0] += 1
                d[b_][1] += str_common_word(b_, d_prod_query[i][1])
                d[b_][2] += str_common_word2(b_, d_prod_query[i][1])
                d[b_][3] =  list(set(d[b_][3] + col_lst))

ds2 = pd.DataFrame.from_dict(d,orient='index')
ds2.columns = ['count','in title 1','in title 2','attribute type']
ds2 = ds2.sort_values(by=['count'], ascending=[False])

f = open("word_review_v2.csv", "w")
f.write("word|count|in title 1|in title 2|attribute type\n")
for i in range(len(ds2)):
    f.write(ds2.index[i] + "|" + str(ds2["count"][i]) + "|" + str(ds2["in title 1"][i]) + "|" + str(ds2["in title 2"][i]) + "|" + str(ds2["attribute type"][i]) + "\n")
f.close()
print("--- File Created: %s minutes ---" % round(((time.time() - start_time)/60),2))