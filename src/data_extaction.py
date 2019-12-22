import xml.etree.ElementTree as ET
from Doc import Doc
import pandas as pd


def extract_xml(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    docs = {}
    for page in root.findall("{http://www.mediawiki.org/xml/export-0.10/}page"):
        id = int(page.find("{http://www.mediawiki.org/xml/export-0.10/}id").text)
        text = page.find("{http://www.mediawiki.org/xml/export-0.10/}revision") \
            .find("{http://www.mediawiki.org/xml/export-0.10/}text").text
        docs[id] = Doc(id, text)
    return docs


def extract_csv(filename):
    docs = {}
    df = pd.read_csv(filename)
    for i in range(df.shape[0]):
        text = df['Title'].values[i] + " " + df['Text'].values[i]
        if "Tag" in df.columns:
            tag = df['Tag'].values[i]
        else:
            tag = None
        docs[i] = Doc(i, text, tag)
    return docs


def read_docs(filename):
    if filename.endswith('xml'):
        return extract_xml(filename)
    return extract_csv(filename)


if __name__ == "__main__":
    # docs = read_docs("../data/Persian.xml")
    docs = read_docs("../data/phase2_train.csv")

    print(docs[0].tag, docs[0].id, docs[0].text)
