import random
from abc import abstractmethod
from math import log

from data_extaction import read_docs
from positional_indexing import PositionalIndexer
from preprocess import EnglishPreprocessor
from query import calc_tfidf


class Classifier:

    @abstractmethod
    def train(self, docs, index):
        pass

    @abstractmethod
    def classify(self, docs):
        pass


class NBClassifier(Classifier):

    def train(self, docs, index):
        self.docs = docs
        self.index = index
        self.tag_wordcount = {}
        self.tag_doccount = {}
        for doc in docs.values():
            tag = doc.tag
            if tag not in self.tag_doccount:
                self.tag_doccount[tag] = 0
            if tag not in self.tag_wordcount:
                self.tag_wordcount[tag] = 0
            self.tag_wordcount[tag] += len(doc.words)
            self.tag_doccount[tag] += 1
        self.prior = {}
        for tag in self.tag_doccount:
            self.prior[tag] = log(self.tag_doccount[tag] / len(docs.keys()))

    def calc_score(self, tag, term):
        term_tagcount = 0
        if term in self.index:
            for posting in self.index[term].postings:
                if self.docs[posting.doc_id].tag == tag:
                    term_tagcount += len(posting.positions)
            return log((term_tagcount + 1) / (self.tag_wordcount[tag] + len(index.keys())))
        return log(1 / (self.tag_wordcount[tag] + len(index.keys()) + 1))


    def classify(self, docs):
        results = {}
        for doc in docs.values():
            scores = []
            for tag in self.prior:
                score = self.prior[tag]
                for word in doc.words:
                    score += self.calc_score(tag, word)
                scores.append((tag, score))
            results[doc.id] = sorted(scores, key=lambda x: -x[1])[0][0]
        return results


class KNNClassifier(Classifier):

    def calc_dist(self, v1, v2):
        dist = sum([(v1[x] - v2[x])**2 for x in v1.keys() & v2.keys()])
        dist += sum([v1[x]**2 for x in v1.keys() - v2.keys()])
        dist += sum([v2[x]**2 for x in v2.keys() - v1.keys()])
        return dist

    def set_k(self, k):
        self.k = k

    def train(self, docs, index):
        self.docs = docs
        self.index = index

    def classify(self, docs):
        results = {}
        for doc in docs.values():
            v_doc = calc_tfidf(doc, self.index, len(self.docs), "ntn")
            knn = []
            for train_doc in self.docs.values():
                v_train_doc = calc_tfidf(train_doc, self.index, len(self.docs), "ntn")
                knn.append((train_doc, self.calc_dist(v_doc, v_train_doc)))
            knn.sort(key=lambda x: x[1])
            knn_tags = [x[0].tag for x in knn[0:self.k]]
            results[doc.id] = max(set(knn_tags), key = knn_tags.count)
        return results

    def __init__(self, k):
        self.k = k


class SVMClassifier(Classifier):

    def train(self, docs, index):
        pass

    def classify(self, docs):
        pass


class RFClassifier(Classifier):

    def train(self, docs, index):
        pass

    def classify(self, docs):
        pass


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


    # classifier = NBClassifier()
    classifier = KNNClassifier(5)

    sampled = {}
    for i in random.sample(train_docs.keys(), 500):
        sampled[i] = train_docs[i]
    classifier.train(sampled, index)
    results = classifier.classify({0: test_docs[0]})


    for i in list(results.keys())[0:1]:
        print("Predicted Tag:", results[i])
        print("Text:", test_docs[i].text)

