import argparse
import os
from collections import OrderedDict, Counter
from operator import itemgetter


"""
This program is designed to extract n-grams (unigram, bigrams, and trigrams) from a segmented text file and 
output them in a specific format.

The output file containing the n-grams, formatted as follows:

- The file is divided into three sections, indicated by the keywords `#unigram`, `#bigrams`, and `#trigrams`.
- Each keyword is followed by the total count of unigram, bigrams, or trigrams extracted.
- Each line in these sections represents an n-gram, followed by its occurrence count in the corpus.

Example format:
```
#unigram 12369
<s> 17073
le 3906
24 2
...
garantirai 1

#bigrams 71090
<s> <s> 8535
<s> le 286
le 24 1
...
helder . 1

#trigrams 135707
<s> <s> le 286
<s> le 24 1
le 24 février 1
...
helder . </s> 1
```

- The unigram section lists individual words with their occurrence counts. - The bi-gram and trigram sections list 
word pairs and triples, respectively, along with how often they appear in the corpus.

To launch this program, you will need to run the following command:
python3 ngram.py ./data/alexandre_dumas/inputFileName.tok
"""


def read_file(input_file):
    # Read the file and split the content into tokens
    with open(input_file, "r", encoding="utf-8-sig", errors="ignore") as f:
        text = f.read()
        # Split the tokenized text on whitespace
        tokens = text.split()
        f.close()
    return tokens


def extract_ngrams(tokens, model):
    extracted_ngrams = zip(*[tokens[i:] for i in range(model)])
    return extracted_ngrams, Counter(extracted_ngrams)


def sort_ngrams(ngrams, reverse=True):
    # Sort n-grams by their occurrence
    sorted_ngrams = OrderedDict(sorted(ngrams.items(), key=itemgetter(1), reverse=reverse))
    return sorted_ngrams


def write_sorted_ngrams_to_file(sorted_ngrams, output_file, ngram_type):
    # Write the header with total count
    output_file.write(f"#{ngram_type}s {len(sorted_ngrams)}\n")
    for ngram, occurrences in sorted_ngrams.items():
        # Write each n-gram and its occurrence to the output file
        output_file.write(' '.join(ngram) + ' ' + str(occurrences) + '\n')


def process_file(input_file):
    # Read the file and split the content into tokens
    tokens = read_file(input_file)

    # Count unigram, bigrams, and trigrams
    _, unigrams = extract_ngrams(tokens, 1)
    _, bigrams = extract_ngrams(tokens, 2)
    _, trigrams = extract_ngrams(tokens, 3)

    # Sort lists of n-grams by its frequency (in descending order by default)
    sorted_unigrams = sort_ngrams(unigrams)
    sorted_bigrams = sort_ngrams(bigrams)
    sorted_trigrams = sort_ngrams(trigrams)

    return sorted_unigrams, sorted_bigrams, sorted_trigrams


def write_file(output_file_name, sorted_unigrams, sorted_bigrams, sorted_trigrams):
    # Open the output file for writing
    with open(output_file_name, "w", encoding="utf-8") as output_file:
        write_sorted_ngrams_to_file(sorted_unigrams, output_file, "unigram")

        # Write the bigrams
        output_file.write("\n")  # New line separator
        write_sorted_ngrams_to_file(sorted_bigrams, output_file, "bigram")

        # Write the trigrams
        output_file.write("\n")  # New line separator
        write_sorted_ngrams_to_file(sorted_trigrams, output_file, "trigram")

    print(f"N-grams written to {output_file_name}")


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Extract n-grams from a segmented text file")
    parser.add_argument("inputFileName", help="Path to the tokenized input file")

    args = parser.parse_args()

    # Get the input file name from the command line argument
    input_file_name = args.inputFileName
    sorted_unigrams, sorted_bigrams, sorted_trigrams = process_file(input_file_name)

    # Create an output file
    output_file_name = f"{os.path.splitext(input_file_name)[0]}_ngrams.txt"
    write_file(output_file_name, sorted_unigrams, sorted_bigrams, sorted_trigrams)

