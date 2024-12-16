import argparse
from perplexity import *


def predict(model, preceding, succeeding):
    """
    This method predicts the missing word based on the context using n-gram probabilities.
    """
    candidates = model.vocabulary  # List of possible candidates (from the corpus)
    predicted_word = max(candidates, key=lambda candidate: model.proba_model.calculate((preceding + candidate)))

    """ predict next word by sliding window when leveraging both the preceding and succeeding"""
    predicted_word = max(candidates, key=lambda candidate: model.proba_model.calculate((preceding + candidate)) * model.proba_model.calculate((candidate + succeeding)))

    return predicted_word


def get_predictions(maskedFileName, model):
    accuracy = 0
    with open(maskedFileName, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split()
            #print(line)
            gap_position = int(line[0]) + 1
            gap = line[gap_position]
            #print("expected: ", gap)
            preceding = tuple(line[gap_position - i] for i in range(model.n_gram_model - 1, 0, -1))
            succeeding = tuple(line[gap_position + i] for i in range(1, model.n_gram_model))
            #print("predicted: ", "".join(predict(model, preceding, succeeding)))
            if gap == "".join(predict(model, preceding, succeeding)):
                accuracy += 1
            #print()
    print("Accuracy: ", accuracy)


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Calculate accuracy of a text filled with n-gram models.")
    parser.add_argument("trainFileName", help="Path to the tokenized (train) input file")
    parser.add_argument("maskedFileName", help="Path to the masked input file")
    parser.add_argument("model", type=int, choices=[1, 2, 3],
                        help="Specify n-gram model (1 for unigram, 2 for bigram, 3 for trigram)")
    parser.add_argument("proba_model", type=str, choices=["MLE", "Laplace", "Kneser-Ney"],
                        help="Specify probability model")
    args = parser.parse_args()

    model = Model(args.model, None)  # Initiate a new model
    model.proba_model = PROBABILITY_MODELS[args.proba_model](model)  #

    model.train(args.trainFileName)  # Train the model with a corpus
    get_predictions(args.maskedFileName, model)
