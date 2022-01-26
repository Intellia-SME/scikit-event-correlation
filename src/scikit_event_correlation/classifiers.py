import time
from math import exp
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.neighbors import KNeighborsMixin, ClassifierMixin, NeighborsBase
from .utils import _check_weights, _get_weights, weighted_mode, _num_samples

def powerset(lst):
    # the power set of the empty set has one element, the empty set
	result = lst
	res = list(map(list, result))
	return res

# k = ageing factor, i = how many steps is the prob active
def linearAgeing(i):
	age = 1 - 0.05*i
	return age

def exponentialAgeing(i):
	k = 0.05
	age = exp(-k*i)
	return age


def updateFreq(event, freq):
	if event not in freq:
		freq[event] = 0
	freq[event] += 1


def updateGraph(prevState, event, graph):
	if event not in graph:
		graph[event] = {}

	if event not in graph[prevState]:
		graph[prevState][event] = 0
	graph[prevState][event] += 1

class EventCorrelationClassifier(KNeighborsMixin, ClassifierMixin, NeighborsBase):
	def __init__(
        self,
        n_neighbors=5,
        *,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
    ):
		super().__init__(
				n_neighbors=n_neighbors,
				algorithm=algorithm,
				leaf_size=leaf_size,
				metric=metric,
				p=p,
				metric_params=metric_params,
				n_jobs=n_jobs,
			)
		self.weights = weights

		def fit(self, X, y):
			"""Fit the k-nearest neighbors classifier from the training dataset.
			Parameters
			----------
			X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
					(n_samples, n_samples) if metric='precomputed'
				Training data.
			y : {array-like, sparse matrix} of shape (n_samples,) or \
					(n_samples, n_outputs)
				Target values.
			Returns
			-------
			self : EventCorrelationClassifier
				The fitted event-correlation classifier.
			"""
			self.weights = _check_weights(self.weights)
			return self._fit(X, y)

		def predict(self, X):
			"""
			Predict the class labels for the provided data.
			Parameters
			----------
			X : array-like of shape (n_queries, n_features), \
					or (n_queries, n_indexed) if metric == 'precomputed'
				Test samples.
			Returns
			-------
			y : ndarray of shape (n_queries,) or (n_queries, n_outputs)
				Class labels for each data sample.
			"""
			neigh_dist, neigh_ind = self.kneighbors(X)
			classes_ = self.classes_
			_y = self._y
			if not self.outputs_2d_:
				_y = self._y.reshape((-1, 1))
				classes_ = [self.classes_]

			n_outputs = len(classes_)
			n_queries = _num_samples(X)
			weights = _get_weights(neigh_dist, self.weights)

			y_pred = np.empty((n_queries, n_outputs), dtype=classes_[0].dtype)
			for k, classes_k in enumerate(classes_):
				if weights is None:
					mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
				else:
					mode, _ = weighted_mode(_y[neigh_ind, k], weights, axis=1)

				mode = np.asarray(mode.ravel(), dtype=np.intp)
				y_pred[:, k] = classes_k.take(mode)

			if not self.outputs_2d_:
				y_pred = y_pred.ravel()

			return y_pred

		def predict_proba(self, X):
			"""Return probability estimates for the test data X.
			Parameters
			----------
			X : array-like of shape (n_queries, n_features), \
					or (n_queries, n_indexed) if metric == 'precomputed'
				Test samples.
			Returns
			-------
			p : ndarray of shape (n_queries, n_classes), or a list of n_outputs \
					of such arrays if n_outputs > 1.
				The class probabilities of the input samples. Classes are ordered
				by lexicographic order.
			"""
			neigh_dist, neigh_ind = self.kneighbors(X)

			classes_ = self.classes_
			_y = self._y
			if not self.outputs_2d_:
				_y = self._y.reshape((-1, 1))
				classes_ = [self.classes_]

			n_queries = _num_samples(X)

			weights = _get_weights(neigh_dist, self.weights)
			if weights is None:
				weights = np.ones_like(neigh_ind)

			all_rows = np.arange(n_queries)
			probabilities = []
			for k, classes_k in enumerate(classes_):
				pred_labels = _y[:, k][neigh_ind]
				proba_k = np.zeros((n_queries, classes_k.size))

				# a simple ':' index doesn't work right
				for i, idx in enumerate(pred_labels.T):  # loop is O(n_neighbors)
					proba_k[all_rows, idx] += weights[:, i]

				# normalize 'votes' into real [0,1] probabilities
				normalizer = proba_k.sum(axis=1)[:, np.newaxis]
				normalizer[normalizer == 0.0] = 1.0
				proba_k /= normalizer

				probabilities.append(proba_k)

			if not self.outputs_2d_:
				probabilities = probabilities[0]

			return probabilities

		def _more_tags(self):
			return {"multilabel": True}