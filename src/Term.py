class Record:
    str = None
    postings = []

    def get_idf(self, total_count):
        pass

    def add_position(self, doc, location):
        pass

    def __init__(self, str, postings = []):
        self.str = str
        self.postings = postings
