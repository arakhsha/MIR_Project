import json
import numpy as np

alpha = float(input("Enter alpha: "))
papers_filename = "../data/semantic_scholar/crawled_papers.json"
papers_file = open(papers_filename, "r")
papers = json.load(papers_file)
papers_file.close()

ids = list(papers.keys())
n = len(papers)
p = np.zeros((n, n))

for paper in papers.values():
    references = paper["references"]
    index = ids.index(paper["id"])
    outs = set(references) & papers.keys()
    if len(outs) == 0:
        row = np.ones(n) / n
    else:
        row = np.zeros(n)
        for ref in outs:
            ref_index = ids.index(ref)
            row[ref_index] = 1/len(outs)
    p[index,] = row

p = p * (1 - alpha)
p = p + (alpha / n) * np.ones((n,n))

pi = np.ones(n) / n
epsilon = 0.00001
while max(np.array(abs(pi - pi @ p))) > epsilon:
    pi = np.array(pi @ p)
print(pi[0:20])