"""
This class is mainly designed to implement the n-gram models and different probability estimators
"""

import ngram as ngram


class ProbabilityModel:
    def __init__(self, model):
        self.model = model

    def calculate(self, m):
        pass


class MLEModel(ProbabilityModel):
    def calculate(self, m):
        numerator_value = self.model.numerator.get(m, 0)
        denominator_value = self.model.denominator.get(m[:self.model.n_gram_model - 1], self.model.N)
        return numerator_value / denominator_value if denominator_value > 0 else 0


class LaplaceModel(ProbabilityModel):
    def calculate(self, m):
        numerator_value = self.model.numerator.get(m, 0) + 1
        if self.model.n_gram_model == 1:
            denominator_value = self.model.N + len(self.model.vocabulary)
        else:
            denominator_value = self.model.denominator.get(m[:self.model.n_gram_model - 1], 0) + len(
                self.model.vocabulary)
        return numerator_value / denominator_value


class KneserNeyModel(ProbabilityModel):
    def __init__(self, model):
        super().__init__(model)
        self.context_dict = {}
        self.continuation_counts = {}
        self.lower_order_model = None

    def preprocess_contexts(self):
        for ngram, count in self.model.numerator.items():
            context = ngram[:-1]
            word = ngram[-1]
            if context not in self.context_dict:
                self.context_dict[context] = []
            self.context_dict[context].append(ngram)

            if word not in self.continuation_counts:
                self.continuation_counts[word] = 0
            self.continuation_counts[word] += 1

    def calculate(self, m, d=0.75):
        if self.context_dict is None:
            self.preprocess_contexts()

        if self.model.n_gram_model == 1:
            return LaplaceModel(self.model).calculate(m)

        context = m[:-1]
        word = m[-1]
        numerator_value = max(self.model.numerator.get(m, 0) - d, 0)

        if numerator_value == 0:
            # Initialization of the lower-order model
            if self.lower_order_model is None:
                self.lower_order_model = Model(self.model.n_gram_model - 1, None)
                self.lower_order_model.proba_model = KneserNeyModel(self.lower_order_model)
                self.lower_order_model.train(self.model.train_corpus)

            return self.lower_order_model.proba_model.calculate(m[1:])
        else:
            denominator_value = self.model.denominator.get(context, 1)
            discounted_prob = numerator_value / denominator_value

            # Continuation probability
            continuation_prob = self.continuation_counts.get(word, 0) / len(self.model.numerator)

            # Back-off weight using precomputed context dictionary
            context_count = len(self.context_dict.get(context, []))
            backoff_weight = (d / denominator_value) * context_count if denominator_value > 0 else 0

        return discounted_prob + backoff_weight * continuation_prob


class Model:
    def __init__(self, n_gram_model, proba_model: ProbabilityModel):
        self.n_gram_model = n_gram_model
        self.denominator = None
        self.numerator = None
        self.V = None
        self.vocabulary = None
        self.N = None
        self.tokens = None
        self.proba_model = proba_model
        self.proba = {}

    def train(self, train_corpus):
        self.train_corpus = train_corpus
        self.tokens = ngram.read_file(train_corpus)
        self.N = len(self.tokens)

        _, self.vocabulary = ngram.extract_ngrams(self.tokens, 1)
        self.V = len(self.vocabulary)

        _, self.numerator = ngram.extract_ngrams(self.tokens, self.n_gram_model)
        # print(f"Number of {self.n_gram_model}-grams : {len(self.numerator)}")
        _, self.denominator = ngram.extract_ngrams(self.tokens, self.n_gram_model - 1)
        # print(f"Number of {self.n_gram_model - 1}-grams : {len(self.denominator)}")

        for n in range(self.n_gram_model - 1, self.N):
            m = tuple(self.tokens[n - i] for i in range(self.n_gram_model - 1, -1, -1))
            self.proba[m] = self.proba_model.calculate(m)
