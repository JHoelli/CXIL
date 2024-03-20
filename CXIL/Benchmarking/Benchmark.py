#TODO inspired by Carla --> Run Through 
import timeit
from typing import List
import pandas as pd
from XIL.Benchmarking.Evaluation import Evaluation
class Benchmark:
    """
    The benchmarking class contains all measurements.
    It is possible to run only individual evaluation metrics or all via one single call.
    For every given factual, the benchmark object will generate one counterfactual example with
    the given recourse method.
    Parameters
    ----------
    mlmodel: carla.models.MLModel
        Black Box model we want to explain.
    recourse_method: carla.recourse_methods.RecourseMethod
        Recourse method we want to benchmark.
    factuals: pd.DataFrame
        Instances for which we want to find counterfactuals.
    """

    def __init__(
        self,
        models,
        explanations,
    ) -> None:
    #TODO What do I need as Init ? 
        self.models=models
        self.explanation = explanations

    def run_benchmark(self, measures: List[Evaluation]) -> pd.DataFrame:
        """
        Runs every measurement and returns every value as dict.
        Parameters
        ----------
        measures : List[Evaluation]
            List of Evaluation measures that will be computed.
        Returns
        -------
        pd.DataFrame
        """
        pipeline = [
            measure.get_evaluation(
                counterfactuals=self._counterfactuals, factuals=self._factuals
            )
            for measure in measures
        ]

        output = pd.concat(pipeline, axis=1)

        return output