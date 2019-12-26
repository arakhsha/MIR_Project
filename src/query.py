import heapq
from collections import Counter
from math import log, sqrt
import random


import pandas

from Doc import Doc
from data_extaction import read_docs
from positional_indexing import PositionalIndexer
from preprocess import EnglishPreprocessor, PersianPreprocessor

from Classifier import RFClassifier, calc_tfidf, slice_index


def classify(docs):
    print("Classifying...")
    print("Preprocessing Train Data...")
    train_docs = read_docs('../data/phase2_train.csv')
    preprocessor = EnglishPreprocessor(train_docs)
    for doc in train_docs.values():
        doc.words = preprocessor.preprocess(doc.text)
    print("Indexing Train Data...")
    index = PositionalIndexer(train_docs, 1).index
    sliced_index = slice_index(index=index, n=200)
    sampled = {}
    sample_size = 500
    for i in random.sample(train_docs.keys(), sample_size):
        sampled[i] = train_docs[i]
    train_docs = sampled
    classifier = RFClassifier(train_docs, sliced_index)
    classifier.train()
    y_pred = classifier.classify(docs)
    doc_ids = [doc.id for doc in docs.values()]
    for i in range(len(doc_ids)):
        docs[doc_ids[i]].tag = y_pred[i]

    # for i in range(10):
    #     doc = docs[doc_ids[i]]
    #     print(doc.id)
    #     print(doc.tag)
    #     print(doc.text)

def calc_diff(v1, v2):
    diff = 0
    for term in set(v1.keys()) & set(v2.keys()):
        diff += v1[term] * v2[term]
    return diff


def search(q_doc, docs, index, result_count, tag=None):
    results = []
    vq = calc_tfidf(q_doc, index, len(docs), "ltc")
    for doc in docs.values():
        if tag is not None:
            if str(doc.tag) != str(tag):
                continue
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

    query_tag = input("Enter Tag (1, 2, 3, 4, None): ")
    tag = None
    if query_tag in ["1", "2", "3", "4"]:
        tag = int(query_tag)

    if tag is not None:
        classify(docs)

    results = search(q_doc, docs, index, 10, query_tag)
    for result in results:
        print(result[1])
        print(result[0].text)
        print()
