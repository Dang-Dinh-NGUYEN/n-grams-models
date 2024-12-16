import argparse
import random
from model import Model
from perplexity import PROBABILITY_MODELS


"""
This program takes as inputs a n-grams model and generates sentences based on the parameters such as :
- the number of sentences to be generated
- the minimum/maximum length of each sentence
"""


def generate(model, sentence_count, max_length=20, min_length=5):
    sentences = []  # To store the generated sentences

    while len(sentences) < sentence_count:
        current_sentence = ["<s>"]  # Start with the beginning-of-sentence token
        count = 0  # Reset word count for the new sentence

        while count < max_length:
            if model.n_gram_model == 1:
                # For unigram, choose the next word randomly from the vocabulary
                next_word = random.choices(
                    list(model.vocabulary.keys()),
                    weights=list(model.vocabulary.values())
                )[0]  # Ensure this is a string
            else:
                # For bigram or trigram, build the context and choose the next word
                context = tuple(current_sentence[-(model.n_gram_model - 1):])  # Get the last n-1 words
                possible_ngrams = {k: v for k, v in model.numerator.items() if k[:len(context)] == context}

                if possible_ngrams:
                    # Select the next n-gram based on the context
                    next_ngram = random.choices(
                        list(possible_ngrams.keys()),
                        weights=list(possible_ngrams.values())
                    )[0]  # This is an n-gram tuple

                    next_word = next_ngram[-1]  # Get the last word from the n-gram tuple
                else:
                    # Fallback to a random word from the vocabulary
                    next_word = random.choices(
                        list(model.vocabulary.keys()),
                        weights=list(model.vocabulary.values())
                    )[0]

            # Ensure next_word is a string before appending
            if isinstance(next_word, tuple):
                next_word = next_word[-1]  # Get the last element if it's a tuple

            current_sentence.append(next_word)  # next_word should be a string
            count += 1

            # Stop if we encounter an end-of-sentence punctuation
            if next_word in ['</s>']:
                break

        # Check if the generated sentence meets the minimum length requirement
        if len(current_sentence) > min_length:
            sentences.append(' '.join(current_sentence))

    return sentences


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate sentences using n-gram models.")
    parser.add_argument("trainFileName", help="Path to the tokenized (train) input file")
    parser.add_argument("model", type=int, choices=[1, 2, 3],
                        help="Specify n-gram model (1 for unigram, 2 for bigram, 3 for trigram)")
    parser.add_argument("sentences", type=int,
                        help="Specify the number of sentences to be generated")
    parser.add_argument("proba_model", type=str, choices=["MLE", "Laplace", "Kneser-Ney"],
                        help="Specify probability model")
    args = parser.parse_args()

    # Initiate and train the model on a training corpus
    model = Model(args.model, None)
    model.proba_model = PROBABILITY_MODELS[args.proba_model](model)  #

    model.train(args.trainFileName)

    generated_sentences = generate(model, args.sentences)
    for sentence in generated_sentences:
        print(sentence)
