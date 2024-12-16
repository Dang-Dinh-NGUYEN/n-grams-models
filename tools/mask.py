import sys
import random

"""
This program takes as input an tokenized text file and select a number of sentences to use for masking purposes.
The program will then add an index number of the beginning of the selected sentence, corresponding to the position of the masked token.

Example : 
    3 <s> <s> le plus grand , le plus fort et le plus adroit surtout est celui qui sait attendre . </s> </s> 
    >> The third token which is 'le' is masked
"""


if len(sys.argv) < 3:
    print("Usage:", sys.argv[0], "tokenized_file", "number_of_sentences_to_be_masked")
    exit(0)

fileName = sys.argv[1]
nbExamples = int(sys.argv[2])

f = open(fileName, "r")
first = True
sentences = []
for line in f:
    tokens = line.split()
    if first:
        padding = 0
        while tokens[padding] == '<s>':
            padding += 1
        first = False
    sentences.append(tokens)

sentNb = len(sentences)
n = 0
alreadySelected = []
while n < nbExamples:
    sentenceIndex = random.randint(0, sentNb - 1)
    tokenPosition = random.randint(padding, len(sentences[sentenceIndex]) - padding - 1)
    tuple = (sentenceIndex, tokenPosition)
    if tuple not in alreadySelected:
        alreadySelected.append(tuple)
        print(tokenPosition, end=" ")
        for token in sentences[sentenceIndex]:
            print(token, end=' ')
        print()
        n += 1
