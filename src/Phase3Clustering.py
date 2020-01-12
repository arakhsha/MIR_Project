from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.mixture import GaussianMixture
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
        vectorizer = TfidfVectorizer(encoding='ISO-8859-1', max_features=100)
        return vectorizer.fit_transform(data).toarray()
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
        return d2v_model.docvecs.vectors_docs


def graph_pca_clustering(matrix, n_clusters, labels, path, special_points=None):
    pca = PCA(n_components=2).fit(matrix)

    datapoint = pca.transform(matrix)

    import matplotlib.pyplot as plt

    color_palette = ["#69D2E7", "#A7DBD8", "#E0E4CC", "#F38630", "#FA6900", "#FE4365", "#FC9D9A", "#F9CDAD", "#C8C8A9",
                     "#83AF9B"]
    label1 = color_palette[:n_clusters]
    color = [label1[i] for i in labels]
    plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color)

    if special_points is not None:
        special_points = pca.transform(special_points)
        plt.scatter(special_points[:, 0], special_points[:, 1], marker='^', s=150, c='#000000')
    plt.savefig(path)
    plt.clf()


def graph_dendrogram(matrix, path):
    from matplotlib import pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage
    import numpy as np
    np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation
    # creating the linkage matrix
    H_cluster = linkage(matrix, 'ward')
    plt.title('Hierarchical Clustering Dendrogram (truncated)')
    plt.xlabel('sample index or (cluster size)')
    plt.ylabel('distance')
    dendrogram(
        H_cluster,
        truncate_mode='lastp',  # show only the last p merged clusters
        leaf_rotation=90.,
        leaf_font_size=12,
        show_contracted=True,  # to get a distribution impression in truncated branches
    )
    plt.savefig(path)
    plt.clf()


def cluster_kmeans(matrix, n_clusters=4):
    kmeans_model = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100)
    X = kmeans_model.fit(matrix)
    return {'labels': kmeans_model.labels_.tolist(),
            'centroids': kmeans_model.cluster_centers_,
            'score': kmeans_model.score(matrix)}


def cluster_GMM(matrix, n_clusters=4):
    gmm_model = GaussianMixture(n_components=n_clusters).fit(matrix)
    return {'labels': gmm_model.predict(matrix),
            'centroids': gmm_model.means_,
            'score': gmm_model.score(matrix)}


def cluster_hierarchical(matrix, n_clusters=4):
    h_model = AgglomerativeClustering(n_clusters=n_clusters).fit(matrix)
    return {'labels': h_model.labels_.tolist(),
            'centroids': None}


if __name__ == "__main__":
    data = pd.read_csv("../Phase3Data/Data.csv", encoding="ISO-8859-1")
    data_text = data['Text'].values

    n_clusters = 4

    vec_methods = ["tf-idf", "word2vec"]
    clust_methods = ["kmeans", "gmm", "hierarchical"]
    for vec_method in vec_methods:

        if vec_method == "tf-idf":
            matrix = vectorize_data(data=data_text, tf_idf_or_word2vec=True)
        else:
            matrix = vectorize_data(data=data_text, tf_idf_or_word2vec=False)

        for clust_method in clust_methods:
            title = vec_method + "_" + clust_method
            print("Doing %s" % title)
            if clust_method == "kmeans":
                result = cluster_kmeans(matrix=matrix, n_clusters=n_clusters)
            elif clust_method == "gmm":
                result = cluster_GMM(matrix=matrix, n_clusters=n_clusters)
            elif clust_method == "hierarchical":
                result = cluster_hierarchical(matrix=matrix, n_clusters=n_clusters)
                graph_dendrogram(matrix, "../Phase3Data/%s_Dendrogram.pdf" % title)

            graph_pca_clustering(matrix=matrix, n_clusters=n_clusters, labels=result['labels'],
                                 path="../Phase3Data/%s_PCA.pdf" % title, special_points=result['centroids'])

            id_label_df = pd.DataFrame({"ID": data["ID"].values, "Text": result["labels"]})
            id_label_df.to_csv(path_or_buf="../Phase3Data/%s.csv" % title, index=False)
