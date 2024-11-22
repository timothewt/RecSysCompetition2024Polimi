from itertools import product
from typing import Type

from tqdm import tqdm

from src.recommender_model import RecommenderModel
from src.utils import open_dataset, train_test_split
from src.utils import evaluate_model


class HyperparametersOptimizer:
	def __init__(self, parameter_space: dict[str: list[float | int | str]], model_class: Type[RecommenderModel]):
		self.parameter_space: dict = parameter_space
		self.model_class = model_class

		self.best_score: float = 0.0
		self.best_parameters: dict = {}

	def _generate_hyperparameters_combinations(self) -> list[dict[str: tuple[float | int | str]]]:
		"""Generates the list of all hyperparameters combinations.
		"""
		keys = list(self.parameter_space.keys())
		values = list(self.parameter_space.values())
		combinations = list(product(*values))
		return [dict(zip(keys, combo)) for combo in combinations]

	def optimize(self) -> tuple[float, dict[str: float | int | str]]:
		parameter_combinations = self._generate_hyperparameters_combinations()

		urm, icm = open_dataset()
		urm_train, urm_test = train_test_split(urm, test_size=0.2)

		model = self.model_class()
		for parameters in (t := tqdm(parameter_combinations, postfix=self._get_summary_string())):
			model.fit(urm=urm_train, icm=icm, urm_val=urm_test, progress_bar=False, **parameters)
			map_10 = evaluate_model(model, urm_test)
			if map_10 > self.best_score:
				self.best_score = map_10
				self.best_parameters = parameters
				t.set_postfix_str(self._get_summary_string())

		return self.best_score, self.best_parameters

	def _get_summary_string(self) -> str:
		return f"Best MAP@10: {self.best_score:.4f} with {[f'{param}: {value:.2e}' for param, value in self.best_parameters.items()]}"
