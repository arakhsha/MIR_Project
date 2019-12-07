class PositionalIndexer:
    index = {}
    docs = []
    gram = None

    def create_index(self, docs, gram):
        pass

    def add_doc(self, doc, gram):
        pass

    def remove_doc(self, id):
        pass

    def get_postings(self, term):
        pass

    def __init__(self, docs, gram):
        self.docs = docs
        self.gram = gram
        self.create_index(docs, gram)

