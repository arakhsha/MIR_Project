import xml.etree.ElementTree as ET

import pandas as pd


class Doc:
    id = None
    text = None

    def __init__(self, id, text):
        self.id = id
        self.text = text


def extract_xml(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    docs = []
    for page in root.findall("{http://www.mediawiki.org/xml/export-0.10/}page"):
        id = page.find("{http://www.mediawiki.org/xml/export-0.10/}id").text
        text = page.find("{http://www.mediawiki.org/xml/export-0.10/}revision")\
                    .find("{http://www.mediawiki.org/xml/export-0.10/}text").text
        docs.append(Doc(id, text))
    return docs


def extract_csv(filename):
    data = pd.read_csv(filename)
    print(data.head(10))


def read_docs(filename):
    if filename.endswith('xml'):
        return extract_xml(filename)
    return extract_csv(filename)


if __name__ == "__main__":
    docs = read_docs("../data/Persian.xml")
    print(docs[0].id, docs[0].text)

    # docs = read_docs("../data/English.csv")
