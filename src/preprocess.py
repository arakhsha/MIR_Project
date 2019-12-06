import nltk
import string
import re
from data_extaction import read_docs


def find_stop_words(self, filename, percent=0.0015):
    docs = read_docs(filename)
    fulltext = ''
    for doc in docs:
        fulltext += doc.text
    tokens = self.remove_punctuation(self.tokenize(self.normalize(fulltext)))
    ft = self.frequency_table(tokens)
    return [x for x, y in ft if y > percent * len(tokens)]


class EnglishPreprocessor:
    stopping_words = []
    stemmer = nltk.PorterStemmer()

    def set_stop_words(self, filename):
        self.stopping_words = find_stop_words(filename)

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

    def preprocess(self, text):
        tokens = self.remove_punctuation(self.tokenize(self.normalize(text)))
        stems = self.stem(self.remove_stop_words(tokens))
        return self.filter_tokens(stems)


if __name__ == "__main__":
    # I'm reading this loud to this kids. These self-identifying kids nowadays read more than I ever did.
    print("Stop Words:", find_stop_words("../data/English.csv"))
    # s = input()
    # ts = preprocess(s)
    # print(ts)
    # print(frequency_table(ts, 3))
