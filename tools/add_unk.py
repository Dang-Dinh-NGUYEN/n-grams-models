#!/usr/bin/env python3

# Replace all tokens that occur less than threshold by <unk>

import sys

corpusFileName = sys.argv[1]
threshold = int(sys.argv[2])
outputFileName = sys.argv[3]

unigram = {}

# Count the frequency of each token in the corpus
with open(corpusFileName, "r", encoding="utf-8") as fi:
    for line in fi:
        tokens = line.split()
        for token in tokens:
            if token not in unigram:
                unigram[token] = 1
            else:
                unigram[token] += 1

# Write the modified corpus to the output file
with open(outputFileName, "w", encoding="utf-8") as fo:
    with open(corpusFileName, "r", encoding="utf-8") as fi:
        for line in fi:
            tokens = line.split()
            for token in tokens:
                if unigram[token] < threshold:
                    fo.write("<unk> ")
                else:
                    fo.write(token + " ")
            fo.write("\n")
