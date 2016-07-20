__author__ = 'mudit'

import re, collections
from configs import *


# def words(text): return re.findall('[a-z]+', text.lower())
def words(text): return text.split(' ')


def train(features):
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model


alphabet = 'abcdefghijklmnopqrstuvwxyz'


def edits1(word):
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
    replaces = [a + c + b[1:] for a, b in splits for c in alphabet if b]
    inserts = [a + c + b for a, b in splits for c in alphabet]
    return set(deletes + transposes + replaces + inserts)


def known_edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)


def known(words): return set(w for w in words if w in NWORDS)


def correct(word):
    candidates = known([word]) or known(edits1(word)) or known_edits2(word) or [word]
    return max(candidates, key=NWORDS.get)


if __name__ == '__main__':
    from load_preprocessed import *

    for query in df_all['search_term'].values[1000:2000]:
        words = query.split()
        for word in words:
            if len(word) > 3 and not word.isdigit():
                cor = correct(word)
                if cor != word:
                    print('%s -> %s | %s' % (query, word, cor))
