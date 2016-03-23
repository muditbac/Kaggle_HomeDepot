# coding=utf-8
__author__ = 'mudit'

from load_data import *
import snowballstemmer
import re
import time
import logging
import enchant

logging.getLogger().setLevel(logging.INFO)

start_time = time.time()
len_train = len(train)
train = pd.concat([train, test])

stemmer = snowballstemmer.stemmer('english')
strNum = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 0}
replace = {'()': '', '-': ' '}

df_brand = attributes[attributes['name'] == "MFG Brand Name"][['product_uid', 'value']].rename(
    columns={"value": "brand"})

i = 0

spell_checker = enchant.Dict('en_US')


def string_preprocess(s, spell_check=False):
    global i
    i += 1
    if i % 10000 == 0:
        logging.info(' %d items processed' % i)
    s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s)  # Split words with a.A
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)  # Split words with aA
    s = s.lower()
    s = s.replace("  ", " ")
    s = re.sub(r"([0-9])( *),( *)([0-9])", r"\1\4", s)
    s = s.replace(",", " ")
    s = s.replace("$", " ")
    s = s.replace("?", " ")
    s = s.replace("-", " ")
    s = s.replace("(", " ")
    s = s.replace(")", " ")
    s = s.replace(":", " ")
    s = s.replace("//", "/")
    s = s.replace("..", ".")
    s = s.replace(" / ", " ")
    s = s.replace(" \\ ", " ")
    s = s.replace(".", " . ")
    s = s.replace("   ", " ")
    s = s.replace("  ", " ").strip(" ")
    s = re.sub(r"(.*)\.$", r"\1", s)  # end period
    s = re.sub(r"(.*)\/$", r"\1", s)  # end period
    s = re.sub(r"^\.(.*)", r"\1", s)  # start period
    s = re.sub(r"^\/(.*)", r"\1", s)  # start slash
    s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
    s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
    s = s.replace(" x ", " xbi ")
    s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
    s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
    s = s.replace("*", " xbi ")
    s = s.replace(" by ", " xbi ")
    s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
    s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1 in. ", s)
    s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1 ft. ", s)
    s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1 lb. ", s)
    s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1 sq.ft. ", s)
    s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1 cu.ft. ", s)
    s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1 gal. ", s)
    s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1 oz. ", s)
    s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1 cm. ", s)
    s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1 mm. ", s)
    s = s.replace("°", " degrees ")
    s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1 deg. ", s)
    s = s.replace(" v ", " volt. ")
    s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1 volt. ", s)
    s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1 watt. ", s)
    s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1 amp. ", s)
    s = s.replace("  ", " ")
    s = s.replace(" . ", " ")
    s = (" ").join([str(strNum[z]) if z in strNum else z for z in s.split(" ")])
    s = s.lower()
    # text = text.decode('utf-8')
    # text = unicodedata.normalize('NFKD',unicode(text.lower())).encode('ascii', 'ignore')
    # for rep in replace.keys():
    #     for rchr in rep:
    #         if rchr in s:
    #             s = s.replace(rchr, replace[rep])
    # if not spell_check:
    #     [spell_checker.add(word) for word in s.split()]
    # else:
    #     words = s.split()
    #     new_s = []
    #     for word in words:
    #         if not spell_checker.check(word):
    #             cword = spell_checker.suggest(word)[0]
    #             # print('\t\t*Spelling Correcter: %s | %s' % (word, cword))
    #             new_s.append(cword)
    #         else:
    #             new_s.append(word)
    #     s = " ".join(new_s)
    #     del new_s, words
    s = " ".join([spell_checker.suggest(word)[0] if not spell_checker.check(word) else word for word in s.split()])
    s = s.lower()
    return s


print('- Processing started %s minutes' % round(((time.time() - start_time) / 60), 2))

final_train = pd.merge(train, descriptions, how="left", on="product_uid")
final_train = pd.merge(final_train, df_brand, how="left", on="product_uid", )
final_train = final_train.fillna('')

print('- Database merged with product_description %s minutes' % round(((time.time() - start_time) / 60), 2))

final_train['product_title'] = [" ".join(stemmer.stemWords(string_preprocess(s).split(" "))) for s in
                                final_train['product_title']]
print('- Product Title Steamed %s minutes' % round(((time.time() - start_time) / 60), 2))
final_train['product_description'] = [" ".join(stemmer.stemWords(string_preprocess(s).split(" "))) for s in
                                      final_train['product_description']]
print('- Product Description Steamed %s minutes' % round(((time.time() - start_time) / 60), 2))
final_train['brand'] = [" ".join(stemmer.stemWords(string_preprocess(s).split(" "))) for s in
                        final_train['brand']]
print('- Brand Name Steamed %s minutes' % round(((time.time() - start_time) / 60), 2))

attributes = attributes.fillna('')
attributes['value'] = [" ".join(stemmer.stemWords(string_preprocess(s).split(" "))) for s in
                       attributes['value']]
print('- All Attributes Steamed %s minutes' % round(((time.time() - start_time) / 60), 2))

final_train['search_term'] = [" ".join(stemmer.stemWords(string_preprocess(s, spell_check=True).split(" "))) for s in
                              final_train['search_term']]
print('- Search Term Steamed and Spell Checked %s minutes' % round(((time.time() - start_time) / 60), 2))

train = final_train[:len_train].fillna('')
test = final_train[len_train:].fillna('')

train.to_csv(INPUT_PATH + "df_train.csv", encoding="ISO-8859-1")
test.to_csv(INPUT_PATH + "df_test.csv", encoding="ISO-8859-1")

attributes.to_csv(INPUT_PATH + "df_attr.csv", encoding="ISO-8859-1")

"""
# TODO Optimize this
corpus = ' '.join(train['product_title'].tolist()).lower()
corpus = corpus.replace("-", ' ')

tokens = word_tokenize(corpus)

# Creating vocab with count
vocab = {}
for token in tokens:
    if token in vocab:
        vocab[token] += 1
    else:
        vocab[token] = 1

# Sorting vocab according to count
sorted_vocab = sorted(vocab.items(), key=operator.itemgetter(1), reverse=True)
"""
