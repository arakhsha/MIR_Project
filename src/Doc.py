class Doc:
    id = None
    text = None
    terms = None

    def __init__(self, id, text):
        self.id = id
        self.text = text