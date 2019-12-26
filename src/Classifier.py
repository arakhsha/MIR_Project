import heapq
import random
from abc import abstractmethod
from collections import Counter

import numpy
from math import log, sqrt

import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score

from data_extaction import read_docs
from positional_indexing import PositionalIndexer
from preprocess import EnglishPreprocessor

from sklearn import svm


def index_to_length(index):
    result = {}
    for word in index:
        result[word] = 0
        for posting in index[word].postings:
            result[word] = result[word] + len(posting.positions)

    return result


def slice_index(index, n=500):
    lengths = index_to_length(index)
    d = Counter(lengths)

    result = {}
    for word, count in d.most_common(n):
        result[word] = index[word]
    return result


def calc_tfidf(doc, index, index_doc_count, method, include_tf_zero=True):
    v = {}
    if include_tf_zero:
        for word in index:
            if word in doc.words:
                t_count = len([x for x in doc.words if x == word])
                if method[0] == "l":
                    tf = log(t_count + 1)
                elif method[0] == "n":
                    tf = t_count
                else:
                    print("Not Supported tf-idf Method!")

                if method[1] == "n":
                    idf = 1
                elif method[1] == "t":
                    doc_freq = 1 if word not in index else len(index[word].postings) + 1
                    idf = log(index_doc_count / doc_freq)
                else:
                    print("Not Supported tf-idf Method!")

                v[word] = tf * idf
            else:
                v[word] = 0
    else:
        for word in set(doc.words).intersection(index):
            t_count = len([x for x in doc.words if x == word])
            if method[0] == "l":
                tf = log(t_count + 1)
            elif method[0] == "n":
                tf = t_count
            else:
                print("Not Supported tf-idf Method!")

            if method[1] == "n":
                idf = 1
            elif method[1] == "t":
                doc_freq = 1 if word not in index else len(index[word].postings) + 1
                idf = log(index_doc_count / doc_freq)
            else:
                print("Not Supported tf-idf Method!")

            v[word] = tf * idf

    if method[2] == "c":
        normalizer = sqrt(sum([x ** 2 for x in v.values()]))
        for word in v.keys():
            v[word] /= normalizer

    return v


def tfidf_matrix(docs, index, index_doc_count, method):
    result = {}
    for key in docs:
        doc = docs[key]
        result[key] = calc_tfidf(doc, index, index_doc_count, method)
    df = pandas.DataFrame(result).fillna(0).transpose()

    df = df.reindex(sorted(df.columns), axis=1)
    return df


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
    def set_param(self, k):
        self.k = k

    def train(self):
        self.train_tf_idf_matrix = tfidf_matrix(docs=self.train_docs,
                                                index=self.train_index,
                                                index_doc_count=self.index_doc_count,
                                                method="ntn")
        pass

    def top_k(self, knn):
        return [x[1][0] for x in heapq.nlargest(self.k, enumerate(knn), key=lambda elem: elem[1][1])]

    def classify(self, query_docs):
        results = []
        for query_doc in query_docs.values():
            query_doc_tf_idf_dict = calc_tfidf(query_doc, self.train_index, self.index_doc_count, "ntn", False)
            current_tf_idf_matrix = self.train_tf_idf_matrix[query_doc_tf_idf_dict.keys()]
            query_doc_vector = numpy.array(list(query_doc_tf_idf_dict.values()))
            knn = []
            for train_doc, train_doc_row in current_tf_idf_matrix.iterrows():
                train_doc_vector = numpy.array(list(train_doc_row))
                knn.append((train_doc, -numpy.linalg.norm(query_doc_vector - train_doc_vector)))

            knn_tags = [self.train_docs[x].tag for x in self.top_k(knn)]
            results.append(max(set(knn_tags), key=knn_tags.count))
        return results

    def __init__(self, train_docs, train_index, index_doc_count, k=3):
        super().__init__(train_docs, train_index, index_doc_count)
        self.k = k
        self.train_tf_idf_matrix = None

    @staticmethod
    def find_best_parameter(train_data, train_index, index_doc_count, validation_data, possible_parameters):
        maximum_accuracy = -1.0
        arg_max_param = None
        validation_true = [doc.tag for doc in validation_data.values()]
        classifier = KNNClassifier(train_data, train_index, index_doc_count)
        for current_param in possible_parameters:
            classifier.set_param(current_param)
            classifier.train()
            validation_pred = classifier.classify(validation_data)
            accuracy = accuracy_score(y_true=validation_true, y_pred=validation_pred)
            print(str(current_param) + ":\t" + str(accuracy))
            if maximum_accuracy < accuracy:
                arg_max_param = current_param
                maximum_accuracy = accuracy

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
        maximum_accuracy = -1.0
        arg_max_param = None
        validation_true = [doc.tag for doc in validation_data.values()]
        classifier = SVMClassifier(train_data, train_index, index_doc_count)
        for current_param in possible_parameters:
            classifier.set_param(current_param)
            classifier.train()
            validation_pred = classifier.classify(validation_data)
            accuracy = accuracy_score(y_true=validation_true, y_pred=validation_pred)
            print(str(current_param) + ":\t" + str(accuracy))
            if maximum_accuracy < accuracy:
                arg_max_param = current_param
                maximum_accuracy = accuracy

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

    while True:
        method_name = input("Select classifier: 1. Naive Bayes 2. k-NN 3. SVM 4. Random Forest 5.exit")
        classifier = None
        if method_name == "1":
            classifier = NBClassifier(train_docs, sliced_index, index_doc_count)
        elif method_name == "2":
            sampled = {}
            sample_size = 2000
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
            print("Confusion Matrix:")
            print(confusion_matrix(y_true=y_true, y_pred=y_pred))
            print("Accuracy:\t" + str(accuracy_score(y_true=y_true, y_pred=y_pred)))
            print("F1 Scores:\t" + str(f1_score(y_true=y_true, y_pred=y_pred, average=None)))
            print("Precision Scores:\t" + str(precision_score(y_true, y_pred, average=None)))
            print("Recall Scores:\t" + str(recall_score(y_true, y_pred, average=None)))
