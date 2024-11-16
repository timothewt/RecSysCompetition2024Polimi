import numpy as np
import scipy.sparse as sp

from src.algorithms import ItemBasedCF, UserBasedCF, CBF
from src.recommender_model import RecommenderModel
from src.utils import tf_idf


class HybridCBFCF(RecommenderModel):
	def __init__(self, ubcf_coeff: float, ibcf_coeff: float, cbf_coeff: float):
		super(HybridCBFCF, self).__init__()
		self.ubcf = UserBasedCF()
		self.ibcf = ItemBasedCF()
		self.cbf = CBF()

		self.ubcf_coeff = ubcf_coeff
		self.ibcf_coeff = ibcf_coeff
		self.cbf_coeff = cbf_coeff

	def fit(self, urm: sp.csr_matrix, icm: sp.csr_matrix, val_urm: sp.csr_matrix, **kwargs) -> None:
		self.urm = urm
		self.icm = tf_idf(icm)

		self.ubcf.fit(urm, icm)
		self.ibcf.fit(urm, icm)
		self.cbf.fit(urm, icm)

	def _get_recommendations_predictions(self, user_id: int) -> np.ndarray:
		return (
			self.ubcf_coeff * self.ubcf._get_recommendations_predictions(user_id) +
			self.ibcf_coeff * self.ibcf._get_recommendations_predictions(user_id) +
			self.cbf_coeff * self.cbf._get_recommendations_predictions(user_id)
        )
