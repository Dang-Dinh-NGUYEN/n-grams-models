import argparse
import math
import time
from model import *

PROBABILITY_MODELS = {"MLE": MLEModel, "Laplace": LaplaceModel, "Kneser-Ney": KneserNeyModel}


def calculate_log_probability(model_instance, corpus):
    # Calculate the log probability for the given probability model
    log_prob = 0
    for n in range(model_instance.model.n_gram_model - 1, len(corpus)):
        m = tuple(corpus[n - i] for i in range(model_instance.model.n_gram_model - 1, -1, -1))
        if model_instance.calculate(m) == 0:
            return float('inf')
        log_prob += math.log(model_instance.calculate(m))
    return log_prob


def calculate_log_perplexity(log_prob, corpus):
    # Calculate the log perplexity based on log probability
    return (-1 / len(corpus)) * log_prob


def calculate_perplexity(log_perplexity):
    # Calculate perplexity from log perplexity
    if log_perplexity == float('-inf'):
        return float('inf')
    return math.exp(log_perplexity)


def calculate_perplexity_on_corpus(model_instance, corpus):
    corpus = ngram.read_file(corpus)
    log_prob = calculate_log_probability(model_instance.proba_model, corpus)
    log_perplexity = calculate_log_perplexity(log_prob, corpus)
    perplexity = calculate_perplexity(log_perplexity)

    return perplexity


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Calculate the perplexity of n-gram models.")
    parser.add_argument("trainFileName", help="Path to the tokenized (train) input file")
    parser.add_argument("testFileName", nargs='?', default=None, help="Path to the tokenized (test) input file")
    parser.add_argument("model", type=int, choices=[1, 2, 3],
                        help="Specify n-gram model (1 for unigram, 2 for bigram, 3 for trigram)")
    parser.add_argument("proba_model", type=str, choices=["MLE", "Laplace", "Kneser-Ney"],
                        help="Specify probability model")
    args = parser.parse_args()

    train_file_name = args.trainFileName
    if args.testFileName is None:
        print("No test file provided -> calculate perplexity on train")

    # Create a new n-gram model
    model = Model(args.model, None)  # Initialize without a probability model first

    # Assign the appropriate probability model to `model.proba_model`
    model.proba_model = PROBABILITY_MODELS[args.proba_model](model)

    start_time = time.time()
    model.train(train_file_name)
    end_time = time.time()
    print(f"Train time : {end_time - start_time}")

    if args.testFileName:
        corpus = ngram.read_file(args.testFileName)
    else:
        corpus = ngram.read_file(train_file_name)

    print(f"Corpus size : {len(corpus)}")
    # Display the n-gram model used
    print(f"Model: {model.n_gram_model}-gram")
    print(f"Corpus Size (N): {model.N}")
    print(f"Vocabulary Size (V): {model.V}")

    # Calculate log probability, log perplexity, and perplexity using `model.proba_model`
    log_prob = calculate_log_probability(model.proba_model, corpus)
    log_perplexity = calculate_log_perplexity(log_prob, corpus)
    perplexity = calculate_perplexity(log_perplexity)

    print("----------------------------------")
    print(f"{args.proba_model} Log Probability: {log_prob}")
    print(f"{args.proba_model} Log Perplexity: {log_perplexity}")
    print(f"{args.proba_model} Perplexity: {perplexity}")
    print()
