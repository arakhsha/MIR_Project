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


def search(query, docs, index):
    pass



if __name__ == "__main__":
    print("Hello!")
