class Term:
    str = None
    postings = []

    def get_idf(self, total_count):
        pass

    def __init__(self, str, postings = []):
        self.str = str
        self.postings = postings
