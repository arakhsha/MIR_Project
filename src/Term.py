class Term:
    str = None
    postings = {}

    def get_idf(self, total_count):
        pass

    def remove_doc(self, doc_id):
        if doc_id in self.postings:
            del self.postings[doc_id]
