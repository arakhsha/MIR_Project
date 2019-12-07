class Record:

    def get_idf(self, total_count):
        pass

    def add_position(self, doc_id, location):
        pass

    def remove_doc(self, doc_id):
        if doc_id in self.postings:
            del self.postings[doc_id]

    def __init__(self, term, postings=None):
        if postings is None:
            postings = {}
        self.term = term
        self.postings = postings
