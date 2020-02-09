import requests
import json
import lxml.html as lh


def write_results(output_filename, papers):
    file = open(output_filename, "w")
    json.dump(papers, file)
    file.close()


def fetch_data(url):
    print("URL:", url)
    response = requests.get(url)
    doc = lh.document_fromstring(response.content)

    paperId = url2id(url)
    title = list(doc.iterfind('.//meta[@name="citation_title"]'))[0].get("content")
    abstract = list(doc.iterfind('.//meta[@name="description"]'))[0].get("content")
    year = list(doc.iterfind('.//meta[@name="citation_publication_date"]'))[0].get("content")
    authors = [x.get("content") for x in doc.iterfind('.//meta[@name="citation_author"]')]
    references = []
    next_in_queue = []

    if len([section for section in doc.iterfind('.//div[@id="references"]')]) > 0:
        references_urls = []
        refernces_section = doc.get_element_by_id("references")
        title_sections = refernces_section.find_class("citation__title")
        for title_section in title_sections:
            reference_id = title_section.get("data-heap-paper-id")
            reference_url = None
            references_links = [link for link in title_section.iterfind('.//a[@href]')]
            if len(references_links) > 0:
                reference_url = "https://www.semanticscholar.org" + references_links[0].get("href")
            if reference_url is not None:
                references.append(reference_id)
                references_urls.append(reference_url)

        references = references[0:10]
        next_in_queue = references_urls[0:5]

    result = {
        "id": paperId,
        "title": title,
        "abstract": abstract,
        "date": year,
        "authors": authors,
        "references": references
    }

    return result, next_in_queue


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
        url = queue.pop(0)
        paperId = url2id(url)
        if paperId not in papers:
            fetched_count += 1
            print("Fetching Paper ", fetched_count)
            data, next_in_queue = fetch_data(url)
            papers[paperId] = data
            queue += next_in_queue
            if fetched_count % 25 == 0:
                print("Saved Progress.")
                save_progress(queue, papers)

            if paperId == "4bd669b70ec1d1fa2d7825bdf7a17ae7c80f4480":
                print(queue)
    return papers


def crawl(start_filename, paper_count):
    star_file = open(start_filename, "r")
    queue = star_file.read().split()
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
    # papers = crawl("../data/semantic_scholar/start.txt", 25)
    queue, papers = read_progress("../data/semantic_scholar/last_queue.json",
                                  "../data/semantic_scholar/last_papers.json")
    # print(len(papers))
    # print(list(papers.values())[0:3])
    papers = continue_progress(queue, papers, 50)
    write_results("../data/semantic_scholar/crawled_papers.json", papers)

    # print(papers)
    # url = "https://www.semanticscholar.org/paper/Tackling-Climate-Change-with-Machine-Learning-Rolnick-Donti/998039a4876edc440e0cabb0bc42239b0eb29644"
    # id = url2id(url)
    # data, next_in_queue = fetch_data(url)
    # print(data)
    # print(next_in_queue)
    # summarized = summarize_data(full_data)
    # print(summarized)
