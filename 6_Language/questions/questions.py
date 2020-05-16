import nltk
import sys
import os
import copy
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    ans = {}
    for filename in os.listdir(directory):
        file = os.path.join(directory, filename)
        f = open(file, "r", encoding="utf8")
        ans[filename] = f.read()
        f.close()

    return ans


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    token = nltk.word_tokenize(document.lower())
    token_copy = copy.copy(token)
    for j in token_copy:
        if j in string.punctuation or j in nltk.corpus.stopwords.words("english"):
            token.remove(j)
    return token


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    words = {}
    numDocs = len(documents)
    for i in documents:
        for j in documents[i]:
            if j not in words:
                num = 0
                for k in documents:
                    if j in documents[k]:
                        num += 1
                words[j] = math.log(numDocs / num)
    return words


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    rank = {}
    for f in files:
        sums = 0
        for q in query:
            if q in files[f]:
                freq = files[f].count(q)
                tfidf = freq * (idfs[q])
                sums += tfidf
        rank[f] = sums
    ans = sorted(rank, key=rank.get, reverse=True)

    return ans[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    rank = {}
    for s in sentences:
        sums = 0
        for q in query:
            if q in sentences[s]:
                sums += (idfs[q])
        rank[s] = sums
    ans = sorted(rank, key=rank.get, reverse=True)

    return ans[:n]


if __name__ == "__main__":
    main()
