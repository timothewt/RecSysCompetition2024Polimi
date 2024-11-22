import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from matplotlib import pyplot as plt

from src.recommender_model import RecommenderModel


def open_dataset() -> tuple[sp.csr_matrix, sp.csr_matrix]:
	"""Opens the dataset (URM and ICM matrices) into sparse matrices

	:return: The URM and ICM as sparse matrices
	:rtype: tuple[sp.csr_matrix, sp.csr_matrix]
	"""
	train = pd.read_csv("../data/data_train.csv")
	icm_metadata = pd.read_csv("../data/data_ICM_metadata.csv")
	urm = sp.csr_matrix((train['data'], (train['user_id'], train['item_id']))).astype(np.float32)
	icm = sp.csr_matrix((icm_metadata['data'], (icm_metadata['item_id'], icm_metadata['feature_id']))).astype(np.float32)
	return urm, icm


def train_test_split(urm: sp.csr_matrix, test_size: float = .2) -> tuple[sp.csr_matrix, sp.csr_matrix]:
	"""Splits the URM matrix into a train and test dataset over the users.

	:param urm: The User-Rating matrix
	:type urm: sp.csr_matrix
	:param test_size: The test size (in [0,1])
	:type test_size: float
	:return: The train and test URM matrices
	:rtype: tuple[sp.csr_matrix, sp.csr_matrix]
	"""
	train_mask = np.random.choice([True, False], urm.getnnz(), p=[1 - test_size, test_size])
	test_mask = ~train_mask

	urm_coo = urm.tocoo()
	urm_train = sp.csr_matrix((urm_coo.data[train_mask], (urm_coo.row[train_mask], urm_coo.col[train_mask])))
	if test_size > 0:
		urm_test = sp.csr_matrix((urm_coo.data[test_mask], (urm_coo.row[test_mask], urm_coo.col[test_mask])))
	else:
		urm_test = sp.csr_matrix([])

	return urm_train, urm_test


def average_precision(recommendations: np.ndarray, y: np.ndarray, k: int = 10) -> float:
	"""Computes the Average Precision of a recommendation

	:param recommendations: Recommendations for a user
	:type recommendations: np.ndarray
	:param y: Ground truth array of relevant items to be recommended
	:type y: np.ndarray
	:param k: Number of items to consider (AP@k)
	:type k: int
	:return: The Average Precision at k for these particular recommendations
	:rtype: float
    """
	relevance_mask = np.isin(recommendations[:k], y)
	precisions = np.cumsum(relevance_mask) / (np.arange(1, k+1))
	return np.sum(precisions * relevance_mask) / min(len(y), k) if len(y) > 0 else 0.


def evaluate_model(trained_model: RecommenderModel, urm_test: sp.csr_matrix, at: int = 10, users_to_test: float = 1.) -> float:
	"""Evaluates a recommender model using the MAP metric

	:param trained_model: A fitted recommender model
	:type trained_model: RecommenderModel
	:param urm_test: The test URM matrix
	:type urm_test: sp.csr_matrix
	:param at: The number of items to recommend to each user
	:type at: int
	:param users_to_test: The ratio of users to test (in [0,1])
	:type users_to_test: float
	:return: The MAP metric for this model on this test data
	:rtype: float
	"""
	cum_ap = 0.
	eval_count = 0

	num_users = urm_test.shape[0]
	users_ids = np.arange(num_users) if users_to_test == 1 else np.random.choice(num_users, size=int(users_to_test * num_users))

	for user_id in users_ids:
		y = urm_test.indices[urm_test.indptr[user_id]:urm_test.indptr[user_id+1]]
		if len(y) > 0:
			eval_count += 1
			recommendations = trained_model.recommend(user_id, at=at)
			cum_ap += average_precision(recommendations, y, k=at)

	return (cum_ap / eval_count).item()


def train_model(model: RecommenderModel, at: int = 10, test_size: float = .2, print_eval: bool = True, **kwargs) -> tuple[RecommenderModel, float]:
	"""Given a recommender model, trains it and evaluates it on test data, then returns the trained model.

	:param model: The model to train, an instance of a recommender model
	:type model: RecommenderModel
	:param at: The number of recommendations given to each user
	:type at: int
	:param test_size: The test size (in [0,1]) for the train/test split. If set to zero, the model uses the whole
	dataset to train and is not evaluated
	:type test_size: float
	:param print_eval: Indicates if the function should print the model evaluation after training
	:type print_eval: bool
	:return: The fitted (trained) recommender model and the MAP@10 score
	:rtype: tuple[RecommenderModel, float]
	"""
	urm, icm = open_dataset()
	urm_train, urm_test = train_test_split(urm, test_size=test_size)

	model.fit(urm=urm_train, icm=icm, urm_val=urm_test, **kwargs)

	map_10 = 0
	if print_eval and test_size > 0:
		map_10 = evaluate_model(model, urm_test, at=at, users_to_test=.2)
		print(f"MAP@{at} evaluation of the {model.__class__.__name__} model: {map_10:.5f}")

	return model, map_10


def write_submission(trained_model: RecommenderModel, filename: str = "submission.csv", at: int = 10) -> None:
	"""Builds the submission file from a trained recommender model. The file is saved in a CSV format.

	:param trained_model: A fitted recommender model
	:type trained_model: RecommenderModel
	:param filename: The filename of the submission for this particular recommender model
	:type filename: str
	:param at: Number of items to recommend
	:type at: int
	"""
	target_users_test = pd.read_csv("../data/data_target_users_test.csv",).to_numpy().ravel()

	recommendations = np.array([
		trained_model.recommend(user_id, at) for user_id in target_users_test
	])

	if not os.path.exists("../submissions"):
		os.makedirs("../submissions")
	with open(f"../submissions/{filename}", "w") as f:
		f.write("user_id,item_list\n")
		for user_id, recs in zip(target_users_test, recommendations):
			f.write(f"{user_id},{' '.join(map(str, recs))}\n")


def tf_idf(mat: sp.csr_matrix) -> sp.csr_matrix:
	"""Rescales the matrix values by weighting the features of the matrix (typically the ICM) using TF-IDF

	:param mat: The sparse matrix
	:type mat: sp.csr_matrix
	:return: The matrix rescaled by TF-IDF
	:rtype: sp.csr_matrix
	"""
	mat = mat.copy()
	df = np.asarray(mat.sum(axis=0)).ravel()
	idf = np.log(mat.shape[0] / (df + 1))
	mat.data = mat.data * idf[mat.tocoo().col]
	mat.eliminate_zeros()
	return mat


def plot_losses(epochs: int, loss_history: np.ndarray | list, loss_history_val: np.ndarray | list = None, num_batch_per_epochs: int = 1, other_data: tuple = None) -> None:
	"""Plots the losses history of a training.

	:param epochs: The number of epochs
	:type epochs: int
	:param loss_history: The loss history
	:type loss_history: np.ndarray | list
	:param loss_history_val: The validation loss history
	:type loss_history_val: np.ndarray | list
	:param num_batch_per_epochs: The number of batches per epoch
	:type num_batch_per_epochs: int
	:param other_data: Other data to plot (optional). The format is (label: str, x: list, y: list)
	:type other_data: tuple
	"""
	plt.plot(loss_history, label="Train loss")
	if loss_history_val is not None:
		plt.plot([x * num_batch_per_epochs for x in range(epochs + 1)], loss_history_val, label="Validation loss")
	plt.xlabel("Train iteration")
	plt.ylabel("Loss")
	plt.title("Loss history")
	plt.legend(loc="upper right")
	if other_data:
		label, x, y = other_data
		ax2 = plt.gca().twinx()
		ax2.plot(x, y, label=label, c="C2")
		plt.legend(loc="lower left")

	plt.grid(True)
	plt.show()