import nltk
import string
import re

from data_extaction import read_docs

stemmer = nltk.PorterStemmer()


def frequency_table(tokens, n=0):
    import collections
    counter = collections.Counter(tokens)
    if n == 0:
        n = len(tokens)
    return counter.most_common(n)


def normalize(text):
    return text.lower()


def tokenize(text):
    return nltk.tokenize.word_tokenize(text)


def remove_punctuation(tokens):
    return [t for t in tokens if t not in string.punctuation]


def remove_stop_words(tokens, stopping_words):
    return [x for x in tokens if x not in stopping_words]


def find_stop_words(filename, percent=0.0004):
    docs = read_docs(filename)
    fulltext = ''
    for doc in docs:
        fulltext += doc.text
    tokens = remove_punctuation(tokenize(normalize(fulltext)))
    ft = frequency_table(tokens)
    return [x for x,y in ft if y > percent*len(tokens)]


def filter_tokens(tokens, min_size=0, special_chars=False):
    if min_size > 0:
        tokens = [t for t in tokens if len(t) >= min_size]
    if special_chars:
        tokens = [t for t in tokens if re.search('[^a-zA-Z-]', t) is None]
    return tokens


def stem(tokens):
    return [stemmer.stem(t) for t in tokens]


def preprocess(text, stopping_words):
    tokens = remove_punctuation(tokenize(normalize(text)))
    stems = stem(remove_stop_words(tokens, stopping_words))
    return filter_tokens(stems)


if __name__ == "__main__":
    # I'm reading this loud to this kids. These self-identifying kids nowadays read more than I ever did.
    print("Stop Words:", find_stop_words("../data/English.csv"))
    # s = input()
    # ts = preprocess(s)
    # print(ts)
    # print(frequency_table(ts, 3))




