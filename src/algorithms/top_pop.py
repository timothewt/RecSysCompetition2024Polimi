import numpy as np
import scipy.sparse as sp

from src.recommender_model import RecommenderModel


class TopPop(RecommenderModel):
	def __init__(self):
		super(TopPop, self).__init__()
		self.urm_train: sp.csr_matrix | None = None
		self.items_popularity: np.ndarray | None = None

	def fit(self, urm: sp.csr_matrix, **kwargs) -> None:
		self.urm = urm
		self.items_popularity = np.ediff1d(urm.tocsc().indptr)

	def _get_recommendations_predictions(self, user_id: int) -> np.ndarray:
		return self.items_popularity
