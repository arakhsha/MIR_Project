from binary_search import binary_search
from Posting import Posting


class Record:
    def get_idf(self, total_count):
        pass

    def add_position(self, doc_id, position):
        index, exists = self.find_doc_index(doc_id)
        if exists:
            self.postings[index].add_position(position)
        else:
            self.postings.insert(index, Posting(doc_id, [position]))

    def find_doc_index(self, doc_id):
        doc_ids = [posting.doc_id for posting in self.postings]
        return binary_search(doc_ids, doc_id)

    def remove_doc(self, doc_id):
        index, exists = self.find_doc_index(doc_id)
        if exists:
            del self.postings[index]

    def __init__(self, term, postings=None):
        if postings is None:
            postings = []
        self.term = term
        self.postings = postings
