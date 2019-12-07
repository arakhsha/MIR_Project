class Posting:
    def __init__(self, doc_id, positions=None):
        if positions is None:
            positions = []
        self.doc_id = doc_id
        self.positions = positions

