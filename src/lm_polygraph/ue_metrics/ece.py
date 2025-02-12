import numpy as np

from typing import List
import netcal.metrics 
from .ue_metric import UEMetric, normalize


class ECE(UEMetric):
    """
    Calculates the Expected Calibration Error (ECE).
    """

    def __init__(self, bins: int = 10, equal_intervals: bool = True):
        """
        Parameters:
            bins (int): Number of bins used for ECE computation.
            equal_intervals (bool): If True, bins have the same width.
        """
        super().__init__()
        self.ece = netcal.metrics.ECE(bins=bins, equal_intervals=equal_intervals)

    def __str__(self):
        return f"ece_{self.ece.bins}" if self.ece.bins != 10 else "ece"

    def __call__(self, estimator: List[float], target: List[float]) -> float:
        """
        Measures the Expected Calibration Error between `estimator` and `target`.

        Parameters:
            estimator (List[float]): A batch of model confidence scores.
            target (List[int]): A batch of ground-truth class labels (0 or 1 for binary classification).

        Returns:
            float: Expected Calibration Error.
        """
        target = np.array(target, dtype=np.float32) 
        estimator = np.array(estimator, dtype=np.float32)
        
        # Normalize target values if needed
        target = normalize(target)
        
        # Compute ECE
        ece_score = self.ece.measure(estimator, target)
        return ece_score
