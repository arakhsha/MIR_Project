import xml.etree.ElementTree as ET

def extract_xml(filename):
    tree = ET.parse(filename)
    namespaces = {'ns': 'http://www.mediawiki.org/xml/export-0.10/'}
    root = tree.getroot()
    docs = []
    for page in root.findall("ns:page", namespaces):
        for text in page.iter("{http://www.mediawiki.org/xml/export-0.10/}text"):
            docs.append(text.text)
        # docs.append([elem.tag for elem in page.iter()])

    return docs



def extract_csv(filename):
    pass


def read_docs(filename):
    if filename.endswith('xml'):
        return extract_xml(filename)
    return extract_csv(filename)


if __name__ == "__main__":
    docs = read_docs("../data/Persian.xml")
    print(docs[1])





