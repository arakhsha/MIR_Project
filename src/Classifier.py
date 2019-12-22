from abc import abstractmethod
from math import log

from data_extaction import read_docs
from positional_indexing import PositionalIndexer
from preprocess import EnglishPreprocessor


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

    def train(self, docs, index):
        pass

    def classify(self, docs):
        pass


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


    classifier = NBClassifier()
    classifier.train(train_docs, index)
    results = classifier.classify(test_docs)

    for i in range(20):
        print("Predicted Tag:", results[i])
        print("Text:", test_docs[i].text)

