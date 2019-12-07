from data_extaction import read_docs
from preprocess import EnglishPreprocessor, PersianPreprocessor

language = input("Select language: 1. English 2. Persian")
if language == "1":
    docs = read_docs('../data/English.csv')
    preprocessor = EnglishPreprocessor(docs)
else:
    docs = read_docs('../data/Persian.xml')
    preprocessor = PersianPreprocessor(docs)

for doc in docs:
    doc.to