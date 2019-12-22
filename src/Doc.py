class Doc:
    def __init__(self, id, text, tag = None):
        self.id = id
        self.text = text
        self.words = None
        self.tag = tag