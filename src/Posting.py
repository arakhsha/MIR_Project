from binary_search import binary_search


class Posting:
    def __init__(self, doc_id, positions=None):
        if positions is None:
            positions = []
        self.doc_id = doc_id
        self.positions = positions

    def add_position(self, position):
        index, exists = binary_search(self.positions, position)
        self.positions.inser(index, position)
