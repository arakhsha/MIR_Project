import itertools
import random
from abc import abstractmethod, ABC
from collections import Counter
from math import log

import numpy
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

    def __init__(self, train_docs, train_index, index_doc_count):
        self.train_docs = train_docs
        self.train_index = train_index
        self.index_doc_count = index_doc_count


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
                # print(posting.doc_id)
                # print([doc_id for doc_id in train_docs])
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

    def __init__(self, train_docs, train_index, index_doc_count):
        super().__init__(train_docs, train_index, index_doc_count)
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
        self.train_tf_idf_matrix = tfidf_matrix(docs=self.train_docs,
                                                index=self.train_index,
                                                index_doc_count=self.index_doc_count,
                                                method="ntn")
        pass

    def top_k(self, knn):
        d = Counter(knn)

        result = []
        for doc, _ in d.most_common(self.k):
            result.append(doc)
        return result

    def classify(self, query_docs):
        results = []
        query_tf_idf_matrix = tfidf_matrix(docs=query_docs,
                                           index=self.train_index,
                                           index_doc_count=self.index_doc_count,
                                           method="ntn")
        for query_doc, query_doc_row in query_tf_idf_matrix.iterrows():
            query_doc_vector = numpy.array(list(query_doc_row))
            knn = {}
            for train_doc, train_doc_row in self.train_tf_idf_matrix.iterrows():
                train_doc_vector = numpy.array(list(train_doc_row))
                knn[train_doc] = -numpy.linalg.norm(query_doc_vector - train_doc_vector)

            knn_tags = [self.train_docs[x].tag for x in self.top_k(knn)]
            results.append(max(set(knn_tags), key=knn_tags.count))
        return results

    def __init__(self, train_docs, train_index, index_doc_count, k=3):
        super().__init__(train_docs, train_index, index_doc_count)
        self.k = k
        self.train_tf_idf_matrix = None

    @staticmethod
    def find_best_parameter(train_data, train_index, index_doc_count, validation_data, possible_parameters):
        maximum_precision = -1.0
        arg_max_param = None
        validation_true = [doc.tag for doc in validation_data.values()]
        classifier = KNNClassifier(train_data, train_index, index_doc_count)
        for current_param in possible_parameters:
            classifier.set_param(current_param)
            classifier.train()
            validation_pred = classifier.classify(validation_data)
            precision = precision_recall_fscore_support(y_true=validation_true,
                                                        y_pred=validation_pred,
                                                        average='macro',
                                                        zero_division=0)[0]
            print(str(current_param) + ":\t" + str(precision))
            if maximum_precision < precision:
                arg_max_param = current_param
                maximum_precision = precision

        return arg_max_param


class SKLearnClassifier(Classifier):

    def train(self):
        self.clf.fit(X=tfidf_matrix(docs=self.train_docs, index=self.train_index,
                                    index_doc_count=self.index_doc_count, method="ntn"),
                     y=[doc.tag for doc in self.train_docs.values()])

    def classify(self, query_docs):
        return self.clf.predict(X=tfidf_matrix(docs=query_docs, index=self.train_index,
                                               index_doc_count=self.index_doc_count, method="ntn"))

    def __init__(self, train_docs, train_index, index_doc_count):
        super().__init__(train_docs, train_index, index_doc_count)
        self.clf = None


class SVMClassifier(SKLearnClassifier):

    def set_param(self, C):
        self.C = C
        self.clf = svm.SVC(C=self.C)

    def __init__(self, train_docs, train_index, index_doc_count, C=1):
        super().__init__(train_docs, train_index, index_doc_count)
        self.C = C
        self.clf = svm.SVC()

    @staticmethod
    def find_best_parameter(train_data, train_index, index_doc_count, validation_data, possible_parameters):
        maximum_precision = -1.0
        arg_max_param = None
        validation_true = [doc.tag for doc in validation_data.values()]
        classifier = SVMClassifier(train_data, train_index, index_doc_count)
        for current_param in possible_parameters:
            classifier.set_param(current_param)
            classifier.train()
            validation_pred = classifier.classify(validation_data)
            precision = precision_recall_fscore_support(y_true=validation_true,
                                                        y_pred=validation_pred,
                                                        average='macro',
                                                        zero_division=0)[0]
            print(str(current_param) + ":\t" + str(precision))
            if maximum_precision < precision:
                arg_max_param = current_param
                maximum_precision = precision

        return arg_max_param


class RFClassifier(SKLearnClassifier):
    def __init__(self, train_docs, train_index, index_doc_count):
        super().__init__(train_docs, train_index, index_doc_count)
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

    index_doc_count = len(train_docs)

    sliced_index = slice_index(index=index, n=200)

    # classifier = NBClassifier()

    # print(sampled)
    while True:
        method_name = input("Select classifier: 1. Naive Bayes 2. k-NN 3. SVM 4. Random Forest 5.exit")
        classifier = None
        if method_name == "1":
            classifier = NBClassifier(train_docs, sliced_index, index_doc_count)
        elif method_name == "2":
            sampled = {}
            sample_size = 1000
            for i in random.sample(train_docs.keys(), sample_size):
                sampled[i] = train_docs[i]
            train_size = int(0.9 * sample_size)
            sliced_train_docs, validation_docs = (dict(list(sampled.items())[:train_size]),
                                                  dict(list(sampled.items())[train_size:]))
            parameter = KNNClassifier.find_best_parameter(sliced_train_docs,
                                                          sliced_index,
                                                          index_doc_count,
                                                          validation_docs,
                                                          [1, 5, 9])
            print("Best parameter is:" + str(parameter))
            classifier = KNNClassifier(sampled, sliced_index, index_doc_count, parameter)
        elif method_name == "3":
            train_size = int(0.9 * len(train_docs))
            sliced_train_docs, validation_docs = (dict(list(train_docs.items())[:train_size]),
                                                  dict(list(train_docs.items())[train_size:]))
            parameter = SVMClassifier.find_best_parameter(sliced_train_docs,
                                                          sliced_index,
                                                          index_doc_count,
                                                          validation_docs,
                                                          [0.5, 1, 1.5, 2])
            print("Best parameter is:" + str(parameter))
            classifier = SVMClassifier(train_docs, sliced_index, index_doc_count, parameter)
        elif method_name == "4":
            classifier = RFClassifier(train_docs, sliced_index, index_doc_count)
        elif method_name == "5":
            exit()
        else:
            print("does not exist")

        if classifier is not None:
            classifier.train()
            y_pred = classifier.classify(test_docs)
            y_true = [doc.tag for doc in test_docs.values()]
            print(confusion_matrix(y_true=y_true, y_pred=y_pred))
