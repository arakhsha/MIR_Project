from Record import Record


class PositionalIndexer:
    def create_index(self):
        for doc in self.docs.values():
            self.add_doc(doc)

    def add_doc(self, doc):
        terms = self.get_terms(doc)
        for i in range(len(terms)):
            term = terms[i]
            if self.index[term] is None:
                self.index[terms] = Record(term, [])
            self.index[term].add_position(doc, i)

    def remove_doc(self, id):
        term_keys = self.get_terms(self.docs[id])
        for term_key in term_keys:
            self.index[term_key].remove_doc(id)

        pass

    def get_postings(self, term):
        return self.index[term].postings

    def get_terms(self, doc):
        words = doc.words
        size = len(words)
        result = []
        for i in range(size-self.gram+1):
            result.append(doc[i: i+self.gram].join(" "))
        return result

    def __init__(self, docs, gram):
        self.index = {}
        self.docs = docs
        self.gram = gram
        self.create_index()

