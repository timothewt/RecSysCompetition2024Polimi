import numpy as np
import scipy.sparse as sp


class RecommenderModel:
	def __init__(self):
		self.urm: sp.csr_matrix | None = None
		self.icm: sp.csr_matrix | None = None
		self.urm_pred: sp.csr_matrix | None = None

	def fit(self, urm: sp.csr_matrix, icm: sp.csr_matrix, urm_val: sp.csr_matrix, progress_bar: bool = True, **kwargs) -> None:
		"""Fits (trains) the model on the given URM and (or) ICM, depending on the algorithm. To be overridden in
		subclasses.

		:param urm: User Ratings Matrix for training
		:type urm: sp.csr_matrix
		:param icm: Item Content Matrix
		:type icm: sp.csr_matrix
		:param urm_val: User Ratings Matrix for validation
		:type urm_val: sp.csr_matrix
		:param progress_bar: If true, progress bar will be shown (if implemented if subclass)
		:type progress_bar: bool
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
		recommendations_predictions = self._get_recommendations_predictions(user_id).astype(np.float32)
		self._exclude_seen_items(user_id, recommendations_predictions)

		top_n_ratings_idx = np.argpartition(-recommendations_predictions, at)[:at]
		top_n_ratings = recommendations_predictions[top_n_ratings_idx]

		return top_n_ratings_idx[
			np.argsort(-top_n_ratings)
		]

	def _get_recommendations_predictions(self, user_id: int) -> np.ndarray:
		"""Gives the recommendations predictions for a given user, which are the probabilities or top-n (the higher,
		the better) that the items should be recommended to the user. It should be overridden in some subclasses

		:param user_id: ID of the user to recommend to
		:type user_id: int
		:return: The recommendations predictions for all the items of the urm
		:rtype: np.ndarray
		"""
		if isinstance(self.urm_pred, sp.spmatrix):
			return self.urm_pred[user_id].toarray().ravel()
		elif isinstance(self.urm_pred, np.ndarray):
			return self.urm_pred[user_id]
		else:
			raise "Unknown type of urm predictions"

	def _exclude_seen_items(self, user_id: int, predicted_ratings: np.ndarray) -> None:
		"""Excludes the items the user has already seen in the predicted ratings list. In-place operation!

		:param user_id: The id of the user
		:type user_id: int
		:param predicted_ratings: The predicted ratings of items for a user
		:type predicted_ratings: np.ndarray
		"""
		seen_items = self.urm.indices[self.urm.indptr[user_id]:self.urm.indptr[user_id + 1]]
		predicted_ratings[seen_items] = -np.inf
