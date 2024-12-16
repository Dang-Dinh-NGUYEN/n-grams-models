import sys
import random

import sys
import random


def load_sentences(corpus_file):
    """Load sentences from the corpus file and return a list of sentences and the total word count."""
    sentence_list = []
    total_words = 0
    with open(corpus_file, 'r', encoding="utf-8-sig", errors="ignore") as fi:
        print(fi)
        for sentence in fi:
            if sentence != "\n":
                sentence_list.append(sentence)
                total_words += len(sentence.split())  # Count words by splitting on whitespace
    return sentence_list, total_words


def split_corpus(sentence_list, total_words, test_ratio):
    """Split the corpus into training and testing based on the test_ratio."""
    sentence_count = len(sentence_list)
    target = int(total_words * test_ratio)
    test_corpus = []
    word_count_in_test = 0

    while word_count_in_test < target:
        n = random.randint(0, sentence_count - 1)
        if sentence_list[n] != "":
            word_count_in_test += len(sentence_list[n].split())
            test_corpus.append(sentence_list[n])
            sentence_list[n] = ""  # Remove the sentence from the original list

    return [sentence for sentence in sentence_list if sentence != ""], test_corpus


def write_corpus(file_name, corpus):
    """Write the corpus to a file."""
    with open(file_name, 'w', encoding="utf-8-sig") as f:
        for sentence in corpus:
            f.write(sentence)


def train_test_split(tokenized_file, ratio):
    train_corpus_file = tokenized_file.replace(".tok", ".train.tok")
    test_corpus_file = tokenized_file.replace(".tok", ".test.tok")

    sentence_list, total_words = load_sentences(tokenized_file)

    train_corpus, test_corpus = split_corpus(sentence_list, total_words, ratio)

    write_corpus(train_corpus_file, train_corpus)
    write_corpus(test_corpus_file, test_corpus)


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage:", sys.argv[0], "tokenized_file", "test_ratio", "train_corpus", "test_corpus")
        exit(0)

    corpus_file = sys.argv[1]
    test_ratio = float(sys.argv[2])
    train_corpus_file = sys.argv[3]
    test_corpus_file = sys.argv[4]

    print("Processing file:", corpus_file)

    # Load sentences and calculate total word count
    sentence_list, total_words = load_sentences(corpus_file)
    print("Corpus size (in tokens):", total_words)

    # Split corpus into training and test sets
    train_corpus, test_corpus = split_corpus(sentence_list, total_words, test_ratio)
    print("Actual size of test corpus (in tokens):", sum(len(s.split()) for s in test_corpus))

    # Write train and test corpus to files
    write_corpus(train_corpus_file, train_corpus)
    write_corpus(test_corpus_file, test_corpus)
