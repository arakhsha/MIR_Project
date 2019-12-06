import nltk
import string
import re

stopwords = nltk.corpus.stopwords.words('english')
stemmer = nltk.PorterStemmer()


def normalize(text):
    return text.lower()


def tokenize(text):
    return nltk.tokenize.word_tokenize(text)


def remove_punctuation(tokens):
    return [t for t in tokens if t not in string.punctuation]


def is_stop_word(token):
    return token.lower() in stopwords


def remove_stop_words(tokens):
    return [t for t in tokens if t.lower() not in stopwords]


def stem(tokens):
    return [stemmer.stem(t) for t in tokens]


def filter_tokens(tokens, min_size=0, special_chars=False):
    if min_size > 0:
        tokens = [t for t in tokens if len(t) >= min_size]
    if special_chars:
        tokens = [t for t in tokens if re.search('[^a-zA-Z-]', t) is None]
    return tokens


def preprocess(text):
    return filter_tokens(stem(remove_stop_words(remove_punctuation(tokenize(normalize(text))))), 3, True)


def frequency_table(tokens, n):
    import collections
    counter = collections.Counter(tokens)
    return counter.most_common(n)


if __name__ == "__main__":
    # I'm reading this loud to this kids. These self-identifying kids nowadays read more than I ever did.
    s = input()
    ts = preprocess(s)
    print(ts)
    print(frequency_table(ts, 3))




