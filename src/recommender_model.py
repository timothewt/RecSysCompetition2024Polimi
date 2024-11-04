class RecommenderModel:
	def __init__(self):
		pass

	def fit(self, *args) -> None:
		raise NotImplementedError

	def recommend(self, user_id: int, at: int = 10):
		raise NotImplementedError
