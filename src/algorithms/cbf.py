import scipy.sparse as sp

from libs.Recommenders.Similarity.Compute_Similarity import Compute_Similarity
from src.recommender_model import RecommenderModel
from src.utils import tf_idf


class CBF(RecommenderModel):
	def __init__(self, top_k: int = 300, shrink: int = 500):
		super(CBF, self).__init__()
		self.similarity_matrix: sp.csr_matrix | None = None
		self.top_k: int = top_k
		self.shrink: int = shrink

	def fit(self, urm: sp.csr_matrix, icm: sp.csr_matrix, urm_val: sp.csr_matrix, **kwargs) -> None:
		self.urm = urm
		self.icm = tf_idf(icm)

		self.similarity_matrix = Compute_Similarity(self.icm.T, topK=min(self.top_k, self.icm.shape[0]), shrink=self.shrink).compute_similarity()

		self.urm_pred = self.urm @ self.similarity_matrix