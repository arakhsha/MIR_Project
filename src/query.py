import heapq
from collections import Counter
from math import log, sqrt

import pandas

from Doc import Doc
from data_extaction import read_docs
from positional_indexing import PositionalIndexer
from preprocess import EnglishPreprocessor, PersianPreprocessor


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
    for word in index:
        t_count = len([x for x in doc.words if x == word])
        if include_tf_zero or t_count > 0:
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


def calc_diff(v1, v2):
    diff = 0
    for term in set(v1.keys()) & set(v2.keys()):
        diff += v1[term] * v2[term]
    return diff


def search(q_doc, docs, index, result_count):
    results = []
    vq = calc_tfidf(q_doc, index, len(docs), "ltc")
    for doc in docs.values():
        vd = calc_tfidf(doc, index, len(docs), "lnc")
        diff = calc_diff(vq, vd)
        results.append((doc, diff))
    results.sort(key=lambda tup: -tup[1])
    return results[0:result_count]


if __name__ == "__main__":
    language = input("INDEX\nSelect language:\n1. English\n2. Persian")

    if language == "1":
        docs = read_docs('../data/English.csv')
        preprocessor = EnglishPreprocessor(docs)
    else:
        docs = read_docs('../data/Persian.xml')
        preprocessor = PersianPreprocessor(docs)

    for doc in docs.values():
        doc.words = preprocessor.preprocess(doc.text)

    print("Preprocess is done!")

    index = PositionalIndexer(docs, 1).index
    print("Index Created Successfully!")

    query = input("Enter Query: ")
    q_doc = Doc(0, query)
    q_doc.words = preprocessor.preprocess(q_doc.text)

    results = search(q_doc, docs, index, 10)
    for result in results:
        print(result[1])
        print(result[0].text)
        print()
