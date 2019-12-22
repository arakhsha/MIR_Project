from math import log, sqrt


def calc_tfidf(doc, index, total_doc_count, method):
    v = {}
    for word in doc.words:
        if method[0] == "l":
            t_count = len([x for x in doc.words if x == word])
            tf = log(t_count + 1)
        else:
            print("Not Supported tf-idf Method!")

        if method[1] == "n":
            idf = 1

        if method[1] == "t":
            idf = log(total_doc_count / len(index[word].postings))

        v[word] = tf * idf

    if method[2] == "c":
        normalizer = sqrt(sum([x^2 for x in v.values()]))
        for word in v.keys():
            v[word] /= normalizer

    return v

def calc_diff(v1, v2):
    diff = 0
    for term in v1.keys().intersection(v2.keys()):
        diff += v1[term] * v2[term]
    return diff


def search(q_doc, docs, index, result_count):
    results = []
    vq = calc_tfidf(q_doc, index, len(docs), "ltc")
    for doc in docs:
        vd = calc_tfidf(doc, index, len(docs), "ltc")
        diff = calc_diff(vq, vd)
        results.append((doc, diff))
    results.sort(key=lambda tup: tup[1])
    return results[0:result_count]




if __name__ == "__main__":
    print("Hello!")
