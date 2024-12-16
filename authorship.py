import os
from model import Model, MLEModel
from perplexity import calculate_perplexity_on_corpus, PROBABILITY_MODELS
import ngram


def info(input_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith(".tok"):
            author = filename.split(".")[0]
            filename = os.path.join(input_dir, filename)

            print(f"Author {author}: {filename}")

            tokens = ngram.read_file(filename)
            print(f"Tokens: {len(tokens)}")
            count, vocab = ngram.extract_ngrams(tokens, 1)
            print(len(vocab))


def train(input_dir, n_gram, estimator):
    models = {}
    perplexity = {}
    for filename in os.listdir(input_dir):
        if filename.endswith(".train.tok"):
            author = filename.split(".")[0]
            train_file = os.path.join(input_dir, filename)

            model = Model(n_gram, estimator)
            model.proba_model = PROBABILITY_MODELS[estimator](model)
            models[author] = model

            model.train(train_file)

            perplexity[author] = calculate_perplexity_on_corpus(model, train_file)

            print(author, ' ', perplexity[author])
    return models, perplexity


def test(input_dir, models, perplexity, top_k=5):
    y_true = []
    y_pred = []
    y_pred_top_k = []  # List to store top k predictions for each test file
    print()
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".test.tok"):
            print(file_name)
            y_true.append(file_name.split(".")[0])
            predictions = {}
            differences = {}
            test_file = os.path.join(input_dir, file_name)
            for author, model in models.items():
                predictions[author] = calculate_perplexity_on_corpus(model, test_file)

                # Calculate the absolute difference from training baseline
                differences[author] = abs(calculate_perplexity_on_corpus(model, test_file) - perplexity[author])

            # Get top k predictions based on lowest perplexity
            sorted_predictions = sorted(predictions.items(), key=lambda item: item[1])[:top_k]
            top_k_authors = [author for author, _ in sorted_predictions]
            print(f"Top {top_k} predictions by Perplexity:", sorted_predictions)
            y_pred_top_k.append(top_k_authors)

            # Get top k predictions based on differences from training baseline
            sorted_differences = sorted(differences.items(), key=lambda item: item[1])[:top_k]
            top_k_authors_2 = [author for author, _ in sorted_differences]
            print(f"Top {top_k} predictions by Difference from Training:", sorted_differences)

            # Use the first predicted author (the one with lowest perplexity) for evaluation
            y_pred.append(top_k_authors_2[0])
            print()
    return y_true, y_pred, y_pred_top_k


def calculate_top_k_accuracy(y_true, y_pred_top_k):
    top_k_accuracy = 0
    for true, top_k_authors in zip(y_true, y_pred_top_k):
        if true in top_k_authors:
            top_k_accuracy += 1
    return top_k_accuracy / len(y_true)


def calculate_mrr(y_true, y_pred_top_k):
    mrr = 0
    for true, top_k_authors in zip(y_true, y_pred_top_k):
        if true in top_k_authors:
            rank = top_k_authors.index(true) + 1  # 1-based rank
            mrr += 1 / rank
    return mrr / len(y_true)


if __name__ == '__main__':
    models, models_perplexity = train("data/authors/", 3, "Kneser-Ney")

    y_true, y_pred, y_pred_top_k = test("data/authors/", models, models_perplexity)

    # Top-k Accuracy
    top_k = 5
    top_k_accuracy = calculate_top_k_accuracy(y_true, y_pred_top_k)
    print()
    print(f"Top-{top_k} Accuracy: {top_k_accuracy}")

    # Mean Reciprocal Rank (MRR)
    mrr = calculate_mrr(y_true, y_pred_top_k)
    print(f"Mean Reciprocal Rank (MRR): {mrr}")
