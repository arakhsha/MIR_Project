from __future__ import unicode_literals

import string

from hazm import *

from preprocess import find_stop_words, remove_stop_words


def normalize(text):
    normalizer = Normalizer()
    return normalizer.normalize(text)


def tokenize(text):
    return word_tokenize(text)


def remove_punctuation(words):
    new_words = []
    punctuations = string.punctuation + '.،:؟!«»؛-[]()'
    for word in words:
        new_word = word
        for punc in list(punctuations):
            new_word = new_word.replace(punc, ' ')
            new_word = new_word.strip()
        if len(new_word) > 0:
            new_words.append(new_word)
    return new_words

def stem(words):
    stemmer = Stemmer()
    new_words = []
    for word in words:
        new_words.append(stemmer.stem(word))
    return new_words


def preprocess(text):
    return stem(remove_stopwords(remove_punctuation(tokenize(normalize(text)))))


if __name__ == "__main__":
    # text = input("Enter Text: ")
    text = 'این نوشته ها، جه بلند و چه کوتاه، بهتر شده اند!'

    text = normalize(text)
    print("Normalized: ", text)

    words = tokenize(text)
    print("Tokenized: ")
    for word in words:
        print(word, end='|')
    print("\n")

    words = remove_punctuation(words)
    print("Without Punctuation: ")
    for word in words:
        print(word, end='|')
    print("\n")

    print("Finding Stopwords...")
    stopwords = find_stop_words("../data/Persian.xml", 0.001)
    print("Without Stopwords:")
    words = remove_stop_words(words, stopwords)
    for word in words:
        print(word, end='|')
    print("\n")

    words = stem(words)
    print("Stemed: ")
    for word in words:
        print(word, end='|')
    print("\n")
