from __future__ import unicode_literals
import nltk
import string
import re
from data_extaction import read_docs

import string

from hazm import *

stemmer = nltk.PorterStemmer()



class EnglishPreprocessor:
    stopping_words = []
    stemmer = nltk.PorterStemmer()

    def find_stop_words(self, filename, percent=0.0015):
        docs = read_docs(filename)
        fulltext = ''
        for doc in docs:
            fulltext += doc.text
        tokens = self.remove_punctuation(self.tokenize(self.normalize(fulltext)))
        ft = self.frequency_table(tokens)
        return [x for x, y in ft if y > percent * len(tokens)]

    def set_stop_words(self, filename):
        self.stopping_words = self.find_stop_words(filename)

    def frequency_table(self, tokens, n=0):
        import collections
        counter = collections.Counter(tokens)
        if n == 0:
            n = len(tokens)
        return counter.most_common(n)

    def normalize(self, text):
        return text.lower()

    def tokenize(self, text):
        return nltk.tokenize.word_tokenize(text)

    def remove_punctuation(self, tokens):
        return [t for t in tokens if t not in string.punctuation]

    def remove_stop_words(self, tokens):
        return [x for x in tokens if x not in self.stopping_words]

    def filter_tokens(self, tokens, min_size=0, special_chars=False):
        if min_size > 0:
            tokens = [t for t in tokens if len(t) >= min_size]
        if special_chars:
            tokens = [t for t in tokens if re.search('[^a-zA-Z-]', t) is None]
        return tokens

    def stem(self, tokens):
        return [stemmer.stem(t) for t in tokens]

    def preprocess(self, text, log=False):
        s = "|"
        if log:
            print("Input:", text)
        text = self.normalize(text)
        if log:
            print("Normalized:", text)
        tokens = self.tokenize(text)
        if log:
            print("Tokens:", s.join(tokens))
        tokens = self.remove_punctuation(tokens)
        if log:
            print("Without Pnctuation:", s.join(tokens))
        tokens = self.remove_stop_words(tokens)
        if log:
            print("Without stopping words:", s.join(tokens))
        stems = self.stem(tokens)
        if log:
            print("Stemmed:", s.join(stems))
        return self.filter_tokens(stems)

    def __init__(self, filename):
        self.set_stop_words(filename)



class PersianPreprocessor:
    stopping_words = []

    def find_stop_words(self, filename, percent=0.0015):
        docs = read_docs(filename)
        fulltext = ''
        for doc in docs:
            fulltext += doc.text
        tokens = self.remove_punctuation(self.tokenize(self.normalize(fulltext)))
        ft = self.frequency_table(tokens)
        return [x for x, y in ft if y > percent * len(tokens)]

    def set_stop_words(self, filename):
        self.stopping_words = self.find_stop_words(filename)

    def frequency_table(self, tokens, n=0):
        import collections
        counter = collections.Counter(tokens)
        if n == 0:
            n = len(tokens)
        return counter.most_common(n)

    def normalize(self, text):
        normalizer = Normalizer()
        return normalizer.normalize(text)

    def tokenize(self, text):
        return word_tokenize(text)

    def remove_punctuation(self, tokens):
        new_words = []
        punctuations = string.punctuation + '.،:؟!«»؛-[]()'
        for word in tokens:
            new_word = word
            for punc in list(punctuations):
                new_word = new_word.replace(punc, ' ')
                new_word = new_word.strip()
            if len(new_word) > 0:
                new_words.append(new_word)
        return new_words

    def remove_stop_words(self, tokens):
        return [x for x in tokens if x not in self.stopping_words]

    def filter_tokens(self, tokens, min_size=0, special_chars=False):
        if min_size > 0:
            tokens = [t for t in tokens if len(t) >= min_size]
        if special_chars:
            tokens = [t for t in tokens if re.search('[^a-zA-Z-]', t) is None]
        return tokens

    def stem(self, tokens):
        stemmer = Stemmer()
        return [stemmer.stem(t) for t in tokens]

    def preprocess(self, text, log=False):
        s = "|"
        if log:
            print("Input:", text)
        text = self.normalize(text)
        if log:
            print("Normalized:", text)
        tokens = self.tokenize(text)
        if log:
            print("Tokens:", s.join(tokens))
        tokens = self.remove_punctuation(tokens)
        if log:
            print("Without Pnctuation:", s.join(tokens))
        tokens = self.remove_stop_words(tokens)
        if log:
            print("Without stopping words:", s.join(tokens))
        stems = self.stem(tokens)
        if log:
            print("Stemmed:", s.join(stems))
        return self.filter_tokens(stems)

    def __init__(self, filename):
        self.set_stop_words(filename)


if __name__ == "__main__":
    # I'm reading this loud to this kids. These self-identifying kids nowadays read more than I ever did.
    # print("Stop Words:", find_stop_words("../data/English.csv"))
    # s = input()
    # ts = preprocess(s)
    # print(ts)
    # print(frequency_table(ts, 3))
    task = input("Select task: 1. Preprocess a text 2. Show frequent words")
    language = input("Select language: 1. English 2. Persian")
    if language == "1":
        preprocessor = EnglishPreprocessor('../data/English.csv')
    else:
        preprocessor = PersianPreprocessor('../data/Persian.xml')

    if task == "1":
        preprocessor.preprocess(input("Enter text:"), True)






