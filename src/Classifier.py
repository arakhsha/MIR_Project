import itertools
import random
from abc import abstractmethod, ABC
from math import log

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from data_extaction import read_docs
from positional_indexing import PositionalIndexer
from preprocess import EnglishPreprocessor
from query import calc_tfidf, tfidf_matrix, slice_index

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
        results = []
        for doc in query_docs.values():
            scores = []
            for tag in self.prior:
                score = self.prior[tag]
                for word in doc.words:
                    score += self.calc_score(tag, word)
                scores.append((tag, score))
            results.append(sorted(scores, key=lambda x: -x[1])[0][0])
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

    def set_param(self, k):
        self.k = k

    def train(self):
        pass

    def classify(self, query_docs):
        results = []
        for doc in query_docs.values():
            v_doc = calc_tfidf(doc, self.train_index, len(self.train_docs), "ntn", False)
            knn = []
            for train_doc in self.train_docs.values():
                v_train_doc = calc_tfidf(train_doc, self.train_index, len(self.train_docs), "ntn", False)
                knn.append((train_doc, self.calc_dist(v_doc, v_train_doc)))
            knn.sort(key=lambda x: x[1])
            knn_tags = [x[0].tag for x in knn[0:self.k]]
            results.append(max(set(knn_tags), key=knn_tags.count))
        return results

    def __init__(self, train_docs, train_index, k=3):
        super().__init__(train_docs, train_index)
        self.k = k

    @staticmethod
    def find_best_parameter(train_data, train_index, validation_data, possible_parameters):
        maximum_precision = -1.0
        arg_max_param = None
        validation_true = [doc.tag for doc in validation_data.values()]
        classifier = KNNClassifier(train_data, train_index)
        for current_param in possible_parameters:
            classifier.set_param(current_param)
            classifier.train()
            validation_pred = classifier.classify(validation_data)
            precision = precision_recall_fscore_support(y_true=validation_true,
                                                        y_pred=validation_pred,
                                                        average='macro',
                                                        zero_division=0)[0]
            print(str(current_param)+":\t"+str(precision))
            if maximum_precision < precision:
                arg_max_param = current_param
                maximum_precision = precision

        return arg_max_param



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

    def set_param(self, C):
        self.C = C
        self.clf = svm.SVC(C=self.C)

    def __init__(self, train_docs, train_index, C=1):
        super().__init__(train_docs, train_index)
        self.C = C
        self.clf = svm.SVC()

    @staticmethod
    def find_best_parameter(train_data, train_index, validation_data, possible_parameters):
        maximum_precision = -1.0
        arg_max_param = None
        validation_true = [doc.tag for doc in validation_data.values()]
        classifier = SVMClassifier(train_data, train_index)
        for current_param in possible_parameters:
            classifier.set_param(current_param)
            classifier.train()
            validation_pred = classifier.classify(validation_data)
            precision = precision_recall_fscore_support(y_true=validation_true,
                                                        y_pred=validation_pred,
                                                        average='macro',
                                                        zero_division=0)[0]
            print(str(current_param)+":\t"+str(precision))
            if maximum_precision < precision:
                arg_max_param = current_param
                maximum_precision = precision

        return arg_max_param


class RFClassifier(SKLearnClassifier):
    def __init__(self, train_docs, train_index):
        super().__init__(train_docs, train_index)
        self.clf = RandomForestClassifier(n_estimators=10)


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

    sliced_index = slice_index(index=index, n=200)

    sampled = {}
    sample_size = 500
    for i in random.sample(train_docs.keys(), sample_size):
        sampled[i] = train_docs[i]

    # classifier = NBClassifier()

    #print(sampled)
    while True:
        method_name = input("Select classifier: 1. Naive Bayes 2. k-NN 3. SVM 4. Random Forest 5.exit")
        classifier = None
        if method_name == "1":
            classifier = NBClassifier(train_docs, sliced_index)
        elif method_name == "2":
            train_size = int(0.9*sample_size)
            train_docs, validation_docs = (dict(list(sampled.items())[:train_size]),
                                           dict(list(sampled.items())[train_size:]))
            parameter = KNNClassifier.find_best_parameter(train_docs, sliced_index, validation_docs, [1, 5, 9])
            print("Best parameter is:"+str(parameter))
            classifier = KNNClassifier(sampled, sliced_index, parameter)
        elif method_name == "3":
            train_size = int(0.9*sample_size)
            train_docs, validation_docs = (dict(list(sampled.items())[:train_size]),
                                           dict(list(sampled.items())[train_size:]))
            parameter = SVMClassifier.find_best_parameter(train_docs, sliced_index, validation_docs, [0.5, 1, 1.5, 2])
            print("Best parameter is:" + str(parameter))
            classifier = SVMClassifier(train_docs, sliced_index, parameter)
        elif method_name == "4":
            train_docs = sampled
            classifier = RFClassifier(train_docs, sliced_index)
        elif method_name == "5":
            exit()
        else:
            print("does not exist")

        if classifier is not None:
            classifier.train()
            y_pred = classifier.classify(train_docs)
            y_true = [doc.tag for doc in train_docs.values()]
            print(confusion_matrix(y_true=y_true, y_pred=y_pred))

