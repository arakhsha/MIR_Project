from Doc import Doc
from Record import Record
from pathlib import Path
import pickle

from data_extaction import read_docs
from preprocess import EnglishPreprocessor, PersianPreprocessor


def unique(list1):
    # intilize a null list
    unique_list = []

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
            # print list
    return unique_list

class PositionalIndexer:
    def create_index(self):
        for doc_id in self.docs.keys():
            self.add_doc(doc_id)

    def add_doc(self, doc_id):
        terms = self.get_terms(doc_id)
        for i in range(len(terms)):
            term = terms[i]
            if term not in self.index:
                self.index[term] = Record(term, [])
            self.index[term].add_position(doc_id, i)

    def remove_doc(self, id):
        term_keys = self.get_terms(self.docs[id])
        for term_key in term_keys:
            self.index[term_key].remove_doc(id)

    def get_record(self, term):
        if term in self.index:
            return self.index[term]
        else:
            return None

    def get_terms(self, doc_id):
        words = self.docs[doc_id].words
        return words

    def __init__(self, docs, gram):
        self.index = {}
        self.docs = docs
        self.gram = gram
        if self.gram == 2:
            all_words = unique([inner
                         for outer in docs
                            for inner in docs[outer].words])
            new_docs = {}
            for i in range(len(all_words)):
                word = "#"+all_words[i]+"#"
                id = i
                words = [word[j:j+2] for j in range(len(word) - 1)]
                doc = Doc(id, ' '.join(words))
                doc.words = words
                new_docs[id] = doc
            self.docs = new_docs

        self.create_index()



def save_index(index, filename):
    outfile = open(filename, 'wb')
    pickle.dump(index, outfile)
    outfile.close()


def load_index(filename):
    infile = open(filename, 'rb')
    index = pickle.load(infile)
    infile.close()
    return index


if __name__ == "__main__":
    # I'm reading this loud to this kids. These self-identifying kids nowadays read more than I ever did.
    # print("Stop Words:", find_stop_words("../data/English.csv"))
    # s = input()
    # ts = preprocess(s)
    # print(ts)
    # print(frequency_table(ts, 3))
    language = input("INDEX\nSelect language:\n1. English\n2. Persian")
    from_saved_files = input("Do you want to read from saved files?\n1. Yes\n2. No")
    ngram = int(input("How many grams?"))

    save_file_address = "../data/saved_" + ("en" if language == "1" else "fa") + "_" + str(ngram) + ".dat"

    index = None
    if from_saved_files != "1":
        if language == "1":
            docs = read_docs('../data/English.csv')
            preprocessor = EnglishPreprocessor(docs)
        else:
            docs = read_docs('../data/Persian.xml')
            preprocessor = PersianPreprocessor(docs)

        for doc in docs.values():
            doc.words = preprocessor.preprocess(doc.text)

        print("Preprocess is done!")

        index = PositionalIndexer(docs, ngram)
        print("Index Created Successfully!")

        save_index(index, save_file_address)
        print("Index Saved Successfully!")
    else:
        if Path(save_file_address).is_file():
            index = load_index(save_file_address)
            print("Index Loaded Successfully!")
        else:
            print("No Saved File Found!")
            exit()

    print({x:{p.doc_id:p.positions for p in index.index[x].postings} for x in index.index})

    # while True:
    #     task = input("What do you want to do?\n"
    #                  "1. Show Posting List for a Term\n"
    #                  "2. Show Positions of a Word in a Document\n"
    #                  "exit. exit")
    #     if task == "exit":
    #         break
    #
    #     record = None
    #     if task == "1" or task == "2":
    #         term = input("Enter Term:")
    #         record = index.get_record(term)
    #         if record is not None:
    #             print([posting.doc_id for posting in record.postings])
    #             if task == "2":
    #                 doc_id = int(input("Enter doc_id:"))
    #                 posting = record.get_posting(doc_id)
    #                 if posting is not None:
    #                     print(posting.positions)
    #                 else:
    #                     print("Posting Not Found!")
    #         else:
    #             print("Term Not Found!")
