import numpy as np
from math import sqrt
from sklearn.base import BaseEstimator, ClassifierMixin #type:ignore
from sklearn.utils.validation import check_array #type:ignore
from numpy.random import default_rng


# --- Distâncias suportadas ---
def euclidean_distance(row1, row2):
    return np.linalg.norm(row1[:-1] - row2[:-1])

def manhattan_distance(row1, row2):
    return np.sum(np.abs(row1[:-1] - row2[:-1]))

def chebyshev_distance(row1, row2):
    return np.max(np.abs(row1[:-1] - row2[:-1]))

def get_distance_function(name):
    if name == 'euclidean':
        return euclidean_distance
    elif name == 'manhattan':
        return manhattan_distance
    elif name == 'chebyshev':
        return chebyshev_distance
    else:
        raise ValueError(f"Unsupported distance metric: {name}")


# --- Seleção do Best Matching Unit ---
def get_best_matching_unit(codebooks, test_row, distance_fn):
    distances = np.array([distance_fn(codebook, test_row) for codebook in codebooks])
    return codebooks[np.argmin(distances)]


# --- Estratégias de Inicialização ---
def random_codebook(train, rng):
    n_records = len(train)
    return train[rng.integers(0, n_records)].copy()

def stratified_mean_codebooks(train, n_codebooks, rng):
    labels = np.unique(train[:, -1])
    codebooks = []
    per_class = n_codebooks // len(labels)
    for label in labels:
        class_samples = train[train[:, -1] == label]
        for _ in range(per_class):
            mean_vector = class_samples[:, :-1].mean(axis=0)
            codebook = np.append(mean_vector, label)
            codebooks.append(codebook)
    while len(codebooks) < n_codebooks:
        codebooks.append(random_codebook(train, rng))  # completa caso divisão não seja exata
    return np.array(codebooks)


# --- Treinamento dos Codebooks ---
def train_codebooks(train, n_codebooks, lrate, epochs, init_strategy, distance_fn, rng):
    if init_strategy == 'random':
        codebooks = np.array([random_codebook(train, rng) for _ in range(n_codebooks)])
    elif init_strategy == 'stratified_mean':
        codebooks = stratified_mean_codebooks(train, n_codebooks, rng)
    else:
        raise ValueError("Invalid init_strategy. Use 'random' or 'stratified_mean'.")

    for epoch in range(epochs):
        rate = lrate * (1.0 - (epoch / float(epochs)))
        for row in train:
            bmu = get_best_matching_unit(codebooks, row, distance_fn)
            for i in range(len(row) - 1):
                error = row[i] - bmu[i]
                if bmu[-1] == row[-1]:
                    bmu[i] += rate * error
                else:
                    bmu[i] -= rate * error
    return codebooks


# --- Classificador LVQ ---
class LVQClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_codebooks=10, lrate=0.1, epochs=100,
                 init_strategy='random', distance_metric='euclidean',
                 random_state=None):
        self.n_codebooks = n_codebooks
        self.lrate = lrate
        self.epochs = epochs
        self.init_strategy = init_strategy
        self.distance_metric = distance_metric
        self.random_state = random_state

    def fit(self, X, y):
        X = check_array(X)
        y = np.array(y)
        train = np.column_stack((X, y))

        self.rng_ = default_rng(self.random_state)
        self.distance_fn_ = get_distance_function(self.distance_metric)

        self.codebooks_ = train_codebooks(
            train,
            self.n_codebooks,
            self.lrate,
            self.epochs,
            self.init_strategy,
            self.distance_fn_,
            self.rng_
        )
        return self

    def predict(self, X):
        X = check_array(X)
        return np.array([
            get_best_matching_unit(self.codebooks_, row, self.distance_fn_)[-1]
            for row in X
        ])

    def predict_proba(self, X):
        X = check_array(X)
        probabilities = []
        for row in X:
            bmu = get_best_matching_unit(self.codebooks_, row, self.distance_fn_)
            label = int(bmu[-1])
            prob = np.zeros(2)
            prob[label] = 1.0
            probabilities.append(prob)
        return np.array(probabilities)

    def decision_function(self, X):
        X = check_array(X)
        return np.array([
            self.distance_fn_(get_best_matching_unit(self.codebooks_, row, self.distance_fn_), row)
            for row in X
        ])
