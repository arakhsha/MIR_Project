import random
from abc import abstractmethod, ABC
from math import log

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from data_extaction import read_docs
from positional_indexing import PositionalIndexer
from preprocess import EnglishPreprocessor
from query import calc_tfidf, tfidf_matrix

from sklearn import svm


class Classifier:

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def classify(self, query_docs):
        pass

    def __init__(self, train_docs, train_index):
        self.train_docs = train_docs
        self.train_index = train_index


class NBClassifier(Classifier):

    def train(self):
        for doc in train_docs.values():
            tag = doc.tag
            if tag not in self.tag_doc_count:
                self.tag_doc_count[tag] = 0
            if tag not in self.tag_word_count:
                self.tag_word_count[tag] = 0
            self.tag_word_count[tag] += len(doc.words)
            self.tag_doc_count[tag] += 1
        for tag in self.tag_doc_count:
            self.prior[tag] = log(self.tag_doc_count[tag] / len(train_docs.keys()))

    def calc_score(self, tag, term):
        term_tag_count = 0
        if term in self.train_index:
            for posting in self.train_index[term].postings:
                if self.train_docs[posting.doc_id].tag == tag:
                    term_tag_count += len(posting.positions)
            return log((term_tag_count + 1) / (self.tag_word_count[tag] + len(index.keys())))
        return log(1 / (self.tag_word_count[tag] + len(index.keys()) + 1))

    def classify(self, query_docs):
        results = {}
        for doc in query_docs.values():
            scores = []
            for tag in self.prior:
                score = self.prior[tag]
                for word in doc.words:
                    score += self.calc_score(tag, word)
                scores.append((tag, score))
            results[doc.id] = sorted(scores, key=lambda x: -x[1])[0][0]
        return results

    def __init__(self, train_docs, train_index):
        super().__init__(train_docs, train_index)
        self.tag_word_count = {}
        self.tag_doc_count = {}
        self.prior = {}


class KNNClassifier(Classifier):

    def calc_dist(self, v1, v2):
        dist = sum([(v1[x] - v2[x]) ** 2 for x in v1.keys() & v2.keys()])
        dist += sum([v1[x] ** 2 for x in v1.keys() - v2.keys()])
        dist += sum([v2[x] ** 2 for x in v2.keys() - v1.keys()])
        return dist

    def set_k(self, k):
        self.k = k

    def train(self):
        pass

    def classify(self, query_docs):
        results = {}
        for doc in query_docs.values():
            v_doc = calc_tfidf(doc, self.train_index, len(self.train_docs), "ntn")
            knn = []
            for train_doc in self.train_docs.values():
                v_train_doc = calc_tfidf(train_doc, self.train_index, len(self.train_docs), "ntn")
                print(v_train_doc)
                knn.append((train_doc, self.calc_dist(v_doc, v_train_doc)))
            knn.sort(key=lambda x: x[1])
            knn_tags = [x[0].tag for x in knn[0:self.k]]
            results[doc.id] = max(set(knn_tags), key=knn_tags.count)
        return results

    def __init__(self, train_docs, train_index, k=3):
        super().__init__(train_docs, train_index)
        self.k = k


class SKLearnClassifier(Classifier):

    def train(self):
        self.clf.fit(X=tfidf_matrix(docs=self.train_docs, index=self.train_index,
                                    total_doc_count=len(self.train_docs), method="ntn"),
                     y=[doc.tag for doc in self.train_docs.values()])

    def classify(self, query_docs):
        return self.clf.predict(X=tfidf_matrix(docs=query_docs, index=self.train_index,
                                               total_doc_count=len(query_docs), method="ntn"))

    def __init__(self, train_docs, train_index):
        super().__init__(train_docs, train_index)
        self.clf = None


class SVMClassifier(SKLearnClassifier):

    def set_C(self, C):
        self.C = C
        self.clf = svm.SVC(C=self.C)

    def __init__(self, train_docs, train_index, C=1):
        super().__init__(train_docs, train_index)
        self.C = C
        self.clf = svm.SVC()


class RFClassifier(SKLearnClassifier):
    def __init__(self, train_docs, train_index):
        super().__init__(train_docs, train_index)
        self.clf = RandomForestClassifier(n_estimators=100)


if __name__ == "__main__":
    train_docs = read_docs('../data/phase2_train.csv')
    test_docs = read_docs('../data/phase2_test.csv')

    preprocessor = EnglishPreprocessor(train_docs)

    for doc in train_docs.values():
        doc.words = preprocessor.preprocess(doc.text)
    for doc in test_docs.values():
        doc.words = preprocessor.preprocess(doc.text)

    print("Preprocess is done!")

    index = PositionalIndexer(train_docs, 1).index
    print("Index Created Successfully!")

    sampled = {}
    for i in random.sample(train_docs.keys(), 500):
        sampled[i] = train_docs[i]

    # classifier = NBClassifier()
    classifier = KNNClassifier(sampled, index, 5)
    classifier.train()
    results = classifier.classify({0: test_docs[0]})

    for i in list(results.keys())[0:1]:
        print("Predicted Tag:", results[i])
        print("Text:", test_docs[i].text)
