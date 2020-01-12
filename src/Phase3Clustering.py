import itertools
from abc import abstractmethod

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import gensim
from gensim.models import Doc2Vec
import numpy as np


def vectorize_data(data, tf_idf_or_word2vec):
    """

    :param data: array of sentences
    :param tf_idf_or_word2vec: True for tf_idf, False for word2doc
    :return: doc x words matrix
    """
    if tf_idf_or_word2vec:
        vectorizer = TfidfVectorizer()
        return vectorizer.fit_transform(data)
    else:
        LabeledSentence1 = gensim.models.doc2vec.TaggedDocument
        all_content_train = []
        j = 0
        for em in data:
            all_content_train.append(LabeledSentence1(em, [j]))
            j += 1
        d2v_model = Doc2Vec(all_content_train, vector_size=100, window=10, min_count=500, workers=7, dm=1,
                            alpha=0.025, min_alpha=0.001)
        d2v_model.train(all_content_train, total_examples=d2v_model.corpus_count, epochs=10, start_alpha=0.002,
                        end_alpha=-0.016)
        return d2v_model.docvecs.doctag_syn0


class KMeans():

    def get_randomized_centroids(self, matrix):
        ids = np.random.permutation(matrix.shape[0])[:self.k]
        return matrix[ids]

    def cluster(self, matrix):
        centroids = self.get_randomized_centroids(matrix)


    def __init__(self, k, max_iter=100):
        self.k = k
        self.max_iter = max_iter


class GMM():

    def cluster(self, matrix, k):
        pass


class Hierarchical():

    def cluster(self, matrix, k):
        pass


if __name__ == "__main__":
    data = pd.read_csv("../Phase3Data/Data.csv", encoding="ISO-8859-1")
    data_text = data['Text'].values()
    print(data.head())
    print(data.columns)

    matrix = vectorize_data(data=data, tf_idf_or_word2vec=True)
    kmeans = KMeans(4)
    print(kmeans.cluster(matrix))

    # Printing the shape of the data and its description
    print(data.shape)
