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
    print("PageRank Results from Iteration")
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
    prob_dist = dict()
    total_pages = len(corpus)

    # Se la pagina corrente non ha link, distribuisce equamente su tutte le pagine
    if len(corpus[page]) == 0:
        for p in corpus:
            prob_dist[p] = 1 / total_pages
        return prob_dist

    # Probabilità base dovuta alla scelta casuale di una pagina qualunque
    random_prob = (1 - damping_factor) / total_pages
    for p in corpus:
        prob_dist[p] = random_prob

    # Aggiunge la probabilità di seguire uno dei link presenti
    links = corpus[page]
    link_prob = damping_factor / len(links)
    for link in links:
        prob_dist[link] += link_prob

    return prob_dist


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    counts = {p: 0 for p in corpus}
    pages = list(corpus.keys())

    # Primo campione scelto uniformemente a caso
    current_page = random.choice(pages)
    counts[current_page] += 1

    # Genera i restanti n - 1 campioni usando la distribuzione del modello di transizione
    for _ in range(n - 1):
        probs = transition_model(corpus, current_page, damping_factor)
        next_pages = list(probs.keys())
        weights = list(probs.values())
        current_page = random.choices(next_pages, weights=weights, k=1)[0]
        counts[current_page] += 1

    # Normalizza i conteggi in probabilità
    return {p: count / n for p, count in counts.items()}


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    total_pages = len(corpus)
    # Condizione iniziale: ogni pagina ha PageRank pari a 1 / N
    pagerank = {p: 1 / total_pages for p in corpus}

    # Crea una copia del corpus dove le pagine senza link puntano a tutte le pagine
    effective_corpus = {}
    for p in corpus:
        if len(corpus[p]) == 0:
            effective_corpus[p] = set(corpus.keys())
        else:
            effective_corpus[p] = corpus[p]

    threshold = 0.001
    random_factor = (1 - damping_factor) / total_pages

    while True:
        new_pagerank = {}
        for p in corpus:
            # Calcola la somma dei contributi da tutte le pagine che puntano a p
            link_sum = 0
            for possible_source in corpus:
                if p in effective_corpus[possible_source]:
                    link_sum += pagerank[possible_source] / len(effective_corpus[possible_source])

            new_pagerank[p] = random_factor + (damping_factor * link_sum)

        # Verifica della convergenza: nessuna variazione deve superare 0.001
        diff = max(abs(new_pagerank[p] - pagerank[p]) for p in corpus)
        pagerank = new_pagerank

        if diff < threshold:
            break

    # Normalizza la somma per garantire che sia esattamente 1
    total_sum = sum(pagerank.values())
    return {p: rank / total_sum for p, rank in pagerank.items()}


if __name__ == "__main__":
    main()
