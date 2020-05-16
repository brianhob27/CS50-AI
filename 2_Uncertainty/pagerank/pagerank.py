import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    links = corpus[page]
    num = (1 - damping_factor) / len(corpus)
    answer = {}

    for key in corpus:
        answer[key] = num

    if len(links):
        temp = damping_factor / len(links)
        for i in links:
            answer[i] = answer[i] + temp
    else:
        temp = 1 / len(corpus)
        for j in answer:
            answer[j] = temp

    return answer


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    answer = {}
    for key in corpus:
        answer[key] = 0
    choice = random.choice(list(corpus.keys()))
    answer[choice] += 1
    num = n - 1

    while num > 0:
        num -= 1
        model = transition_model(corpus, choice, damping_factor)
        choice = random.choices(list(model.keys()), list(model.values()))[0]
        answer[choice] += 1

    for k in answer:
        answer[k] = answer[k] / n

    return answer


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    n = len(corpus)
    num = 1 / n
    answer = {}
    for keys in corpus:
        answer[keys] = num

    changed = True
    while changed:
        changed = False
        for p in corpus:
            summation = 0
            for i in corpus:
                if p != i:
                    if len(corpus[i]):
                        for page in corpus[i]:
                            if page == p:
                                summation = summation + (answer[i] / len(corpus[i]))
                                break
                    else:
                        summation += (answer[i] / n)
            pr = ((1 - damping_factor) / n) + (damping_factor * summation)
            if (pr - answer[p]) > 0.001 or (answer[p] - pr) > 0.001:
                changed = True
                answer[p] = pr
                
    return answer


if __name__ == "__main__":
    main()
