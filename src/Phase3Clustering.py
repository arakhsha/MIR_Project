import itertools
from abc import abstractmethod

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import gensim
from gensim.models import Doc2Vec
from sklearn.decomposition import PCA



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


def cluster_kmeans(self, matrix):
    kmeans_model = KMeans(n_clusters=4, init='k-means++', max_iter=100)
    X = kmeans_model.fit(matrix)
    labels = kmeans_model.labels_.tolist()
    l = kmeans_model.fit_predict(matrix)
    pca = PCA(n_components=2).fit(matrix)
    datapoint = pca.transform(matrix)


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
