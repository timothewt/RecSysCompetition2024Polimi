import os

import numpy as np
import pandas as pd
import scipy.sparse as sp


class RecommenderModel:
	def __init__(self):
		pass

	def fit(self, *args) -> None:
		raise NotImplementedError

	def recommend(self, user_id: int, *args):
		raise NotImplementedError


def open_data() -> tuple[sp.csr_matrix, sp.csr_matrix]:
	train = pd.read_csv("../data/data_train.csv")
	ICM_metadata = pd.read_csv("../data/data_ICM_metadata.csv")
	urm = sp.csr_matrix((train['data'], (train['user_id'], train['item_id'])))
	icm = sp.csr_matrix((ICM_metadata['data'], (ICM_metadata['item_id'], ICM_metadata['feature_id'])))
	return urm, icm


def get_target_users_test() -> np.ndarray:
	return pd.read_csv("../data/data_target_users_test.csv",).to_numpy().ravel()


def make_submission(trained_model: RecommenderModel, filename: str = "submission.csv") -> None:
	target_users_test = get_target_users_test()
	recommendations = np.array([
			trained_model.recommend(user_id) for user_id in target_users_test
	])
	if not os.path.exists("../submissions"):
		os.makedirs("../submissions")
	with open(f"../submissions/{filename}", "w") as f:
		f.write("user_id,item_list\n")
		for user_id, recs in zip(target_users_test, recommendations):
			f.write(f"{user_id},{' '.join(map(str, recs))}\n")
