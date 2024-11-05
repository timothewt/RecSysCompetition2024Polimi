import numpy as np
import scipy.sparse as sp


class RecommenderModel:
	def __init__(self):
		self.urm: sp.csr_matrix | None = None
		self.icm: sp.csr_matrix | None = None
		self.urm_pred: sp.csr_matrix | None = None

	def fit(self, urm: sp.csr_matrix, icm: sp.csr_matrix) -> None:
		"""Fits (trains) the model on the given URM and (or) ICM, depending on the algorithm. To be overridden in
		subclasses.

		:param urm: User Rating Matrix
		:type urm: sp.csr_matrix
		:param icm: Item Content Matrix
		:type icm: sp.csr_matrix
		"""
		raise NotImplementedError

	def recommend(self, user_id: int, at: int = 10) -> np.ndarray:
		"""Gives the top {at} recommended items for this user.

		:param user_id: ID of the user to recommend to
		:type user_id: int
		:param at: Number of items to recommend
		:type at: int
		:return: The {at} most relevant recommended items
		:rtype: np.ndarray
		"""
		recommendations_predictions = self._get_recommendations_predictions(user_id)
		recommended_items = np.argsort(-recommendations_predictions)

		return self._exclude_seen_items(user_id, recommended_items)[:at]

	def _get_recommendations_predictions(self, user_id: int) -> np.ndarray:
		"""Gives the recommendations predictions for a given user, which are the probabilities or top-n (the higher,
		the better) that the items should be recommended to the user. It should be overridden in some subclasses

		:param user_id: ID of the user to recommend to
		:type user_id: int
		:return: The recommendations predictions for all the items of the urm
		:rtype: np.ndarray
		"""
		return self.urm_pred[user_id].toarray().ravel()

	def _exclude_seen_items(self, user_id: int, recommended_items: np.ndarray) -> np.ndarray:
		"""Excludes the items the user has already seen in a recommendations list

		:param user_id: The id of the user
		:type user_id: int
		:param recommended_items: The original list of recommended items to the user
		:type recommended_items: np.ndarray
		:return: The list of recommended items without previously seen items
		:rtype: np.ndarray
		"""
		seen_items = self.urm.indices[self.urm.indptr[user_id]:self.urm.indptr[user_id + 1]]
		unseen_items_mask = np.isin(recommended_items, seen_items, assume_unique=True, invert=True)

		return recommended_items[unseen_items_mask]

