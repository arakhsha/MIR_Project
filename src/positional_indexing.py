from Record import Record


class PositionalIndexer:
    index = {}
    docs = []
    gram = None

    def create_index(self):
        self.index = {}
        for doc in self.docs:
            self.add_doc(doc)

    def add_doc(self, doc):
        terms = self.get_terms(doc)
        for i in range(len(terms)):
            term = terms[i]
            if self.index[term] is None:
                self.index[terms] = Record(term, [])
            self.index[term].add_position(doc, i)

    def remove_doc(self, id):
        pass

    def get_postings(self, term):
        pass

    def get_terms(self, doc):
        words = doc.words
        size = len(words)
        result = []
        for i in range(size-self.gram+1):
            result.append(doc[i: i+self.gram].join(" "))
        return result

    def __init__(self, docs, gram):
        self.docs = docs
        self.gram = gram
        self.create_index()

