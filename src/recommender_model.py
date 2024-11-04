import scipy.sparse as sp


class RecommenderModel:
	def __init__(self):
		pass

	def fit(self, urm_train: sp.csr_matrix, icm: sp.csr_matrix) -> None:
		raise NotImplementedError

	def recommend(self, user_id: int, at: int = 10):
		raise NotImplementedError
