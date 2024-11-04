import os

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.recommender_model import RecommenderModel


def open_dataset() -> tuple[sp.csr_matrix, sp.csr_matrix]:
	"""Opens the dataset (URM and ICM matrices) into sparse matrices

	:return: The URM and ICM as sparse matrices
	:rtype: tuple[sp.csr_matrix, sp.csr_matrix]
	"""
	train = pd.read_csv("../data/data_train.csv")
	icm_metadata = pd.read_csv("../data/data_ICM_metadata.csv")
	urm = sp.csr_matrix((train['data'], (train['user_id'], train['item_id'])))
	icm = sp.csr_matrix((icm_metadata['data'], (icm_metadata['item_id'], icm_metadata['feature_id'])))
	return urm, icm


def train_test_split(urm: sp.csr_matrix, test_size: float = .20) -> tuple[sp.csr_matrix, sp.csr_matrix]:
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
	urm_test = sp.csr_matrix((urm_coo.data[test_mask], (urm_coo.row[test_mask], urm_coo.col[test_mask])))

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


def evaluate_model(trained_model: RecommenderModel, urm_test: sp.csr_matrix, at: int = 10) -> float:
	"""Evaluates a recommender model using the MAP metric

	:param trained_model: A fitted recommender model
	:type trained_model: RecommenderModel
	:param urm_test: The test URM matrix
	:type urm_test: sp.csr_matrix
	:param at: The number of items to recommend to each user
	:type at: int
	:return: The MAP metric for this model on this test data
	:rtype: float
	"""
	cum_ap = 0.
	eval_count = 0

	for user_id in range(urm_test.shape[0]):
		y = urm_test.indices[urm_test.indptr[user_id]:urm_test.indptr[user_id+1]]
		if len(y) > 0:
			eval_count += 1
			recommendations = trained_model.recommend(user_id, at=at)
			cum_ap += average_precision(recommendations, y, k=at)

	return cum_ap / eval_count


def train_and_test_model(model: RecommenderModel, at: int = 10, test_size: float = .2) -> RecommenderModel:
	"""Given a recommender model, trains it and evaluates it on test data, then returns the trained model.

	:param model: The model to train, an instance of a recommender model
	:type model: RecommenderModel
	:param at: The number of recommendations given to each user
	:type at: int
	:param test_size: The test size (in [0,1]) for the train/test split
	:type test_size: float
	:return: The fitted (trained) recommender model
	:rtype: RecommenderModel
	"""
	urm, icm = open_dataset()
	urm_train, urm_test = train_test_split(urm, test_size=test_size)

	model.fit(urm_train, icm)

	print(f"Final evaluation of the {model.__class__.__name__} model: {evaluate_model(model, urm_test, at=at):.5f}")

	return model


def write_submission(trained_model: RecommenderModel, filename: str = "submission.csv") -> None:
	"""Builds the submission file from a trained recommender model. The file is saved in a CSV format.

	:param trained_model: A fitted recommender model
	:type trained_model: RecommenderModel
	:param filename: The filename of the submission for this particular recommender model
	:type filename: str
	"""
	target_users_test = pd.read_csv("../data/data_target_users_test.csv",).to_numpy().ravel()
	recommendations = np.array([
		trained_model.recommend(user_id) for user_id in target_users_test
	])
	if not os.path.exists("../submissions"):
		os.makedirs("../submissions")
	with open(f"../submissions/{filename}", "w") as f:
		f.write("user_id,item_list\n")
		for user_id, recs in zip(target_users_test, recommendations):
			f.write(f"{user_id},{' '.join(map(str, recs))}\n")
