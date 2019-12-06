class Posting:
    doc_id = None
    positions = None


class Indexer:
    index = None
    docs = None

    def create_index(filename, language):
        pass

    def add_doc(self, id, text):
        pass

    def remove_doc(self, id):
        pass

    def get_postings(self, term):
        pass

