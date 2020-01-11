from urllib import request
import json


def write_results(output_filename, papers):
    file = open(output_filename, "w")
    json.dump(papers, file)
    file.close()


def summarize_data(full_data):
    paperId = full_data["paperId"]
    title = full_data["title"]
    abstract = full_data["abstract"]
    year = full_data["year"]
    authors = [x["name"] for x in full_data["authors"]]
    references = [x["paperId"] for x in full_data["references"]][0:10]
    result = {
        "id": paperId,
        "title": title,
        "abstract": abstract,
        "date": year,
        "authors": authors,
        "references": references
    }
    return result


def fetch_full_data(id):
    respone = request.urlopen("http://api.semanticscholar.org/v1/paper/" + id)
    print("URL:", "http://api.semanticscholar.org/v1/paper/" + id)
    data = json.loads(respone.read().decode())
    return data


def url2id(url):
    id = url.split("/")[-1]
    return id


def save_progress(queue, papers):
    queue_file = open("../data/semantic_scholar/last_queue.json", "w")
    json.dump(queue, queue_file)
    queue_file.close()

    papers_file = open("../data/semantic_scholar/last_papers.json", "w")
    json.dump(papers, papers_file)
    papers_file.close()


def continue_progress(queue, papers, paper_count):
    fetched_count = len(papers)
    while fetched_count < paper_count and len(queue) > 0:
        paperId = queue.pop(0)
        if paperId not in papers:
            fetched_count += 1
            print("Fetching Paper ", fetched_count)
            full_data = fetch_full_data(paperId)
            summarized = summarize_data(full_data)
            papers[paperId] = summarized
            queue += summarized["references"][0:5]
            if fetched_count % 25 == 0:
                print("Saved Progress.")
                save_progress(queue, papers)

    return papers


def crawl(start_filename, paper_count):
    star_file = open(start_filename, "r")
    start_urls = star_file.read().split()
    queue = [url2id(url) for url in start_urls]
    papers = {}
    papers = continue_progress(queue, papers, paper_count)
    return papers


def read_progress(queue_filename, papers_filename):
    queue_file = open(queue_filename, "r")
    queue = json.load(queue_file)
    queue_file.close()

    papers_file = open(papers_filename, "r")
    papers = json.load(papers_file)
    papers_file.close()

    return queue, papers


if __name__ == "__main__":
    # papers = crawl("../data/semantic_scholar/start.txt", 5000)
    queue, papers = read_progress("../data/semantic_scholar/last_queue.json",
                                  "../data/semantic_scholar/last_papers.json")
    # print(len(papers))
    # print(list(papers.values())[0:3])
    papers = continue_progress(queue, papers, 5000)
    write_results("../data/semantic_scholar/crawled_papers.json", papers)

    # print(papers)
    # url = "https://www.semanticscholar.org/paper/Tackling-Climate-Change-with-Machine-Learning-Rolnick-Donti/998039a4876edc440e0cabb0bc42239b0eb29644"
    # id = url2id(url)
    # full_data = fetch_full_data(id)
    # summarized = summarize_data(full_data)
    # print(summarized)
