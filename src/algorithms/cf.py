import scipy.sparse as sp

from libs.Recommenders.Similarity.Compute_Similarity import Compute_Similarity
from src.recommender_model import RecommenderModel


class UserBasedCF(RecommenderModel):
	def __init__(self, top_k: int = 300, shrink: int = 500):
		super(UserBasedCF, self).__init__()
		self.similarity_matrix: sp.csr_matrix | None = None
		self.top_k: int = top_k
		self.shrink: int = shrink

	def fit(self, urm: sp.csr_matrix, **kwargs) -> None:
		self.urm = urm

		self.similarity_matrix = Compute_Similarity(self.urm.T, topK=min(self.top_k, self.urm.shape[0]), shrink=self.shrink).compute_similarity()

		self.urm_pred = self.similarity_matrix @ self.urm


class ItemBasedCF(RecommenderModel):
	def __init__(self, top_k: int = 300, shrink: int = 500):
		super(ItemBasedCF, self).__init__()
		self.similarity_matrix: sp.csr_matrix | None = None
		self.top_k: int = top_k
		self.shrink: int = shrink

	def fit(self, urm: sp.csr_matrix, **kwargs) -> None:
		self.urm = urm

		self.similarity_matrix = Compute_Similarity(self.urm, topK=min(self.top_k, self.urm.shape[1]), shrink=self.shrink).compute_similarity()

		self.urm_pred = self.urm @ self.similarity_matrix
