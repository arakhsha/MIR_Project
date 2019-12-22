from abc import abstractmethod


class Classifier:

    @abstractmethod
    def train(self, docs):
        pass

    @abstractmethod
    def classify(self, docs):
        pass


class NBClassifier(Classifier):

    def train(self, docs):
        pass

    def classify(self, docs):
        pass


class KNNClassifier(Classifier):

    def train(self, docs):
        pass

    def classify(self, docs):
        pass


class SVMClassifier(Classifier):

    def train(self, docs):
        pass

    def classify(self, docs):
        pass


class RFClassifier(Classifier):

    def train(self, docs):
        pass

    def classify(self, docs):
        pass


if __name__ == "__main__":
    print("Hi!")
