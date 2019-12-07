class Posting:
    doc_id = None
    positions = []


class PositionalIndexer:
    index = None
    docs = None

    def create_index(docs):
        pass

    def add_doc(self, doc):
        pass

    def remove_doc(self, id):
        pass

    def get_postings(self, term):
        pass

