import numpy as np
from typing import List
from sklearn.metrics import roc_auc_score
from yellowbrick.classifier import ROCAUC
from .ue_metric import UEMetric, normalize, skip_target_nans

class AUROC(UEMetric):
    """
    Calculates the Area Under the Receiver Operating Characteristic Curve (AUROC).
    """

    def __init__(self
                #  , average: str = 'macro', multi_class: str = 'raise'
                 ):
        """
        Parameters:
            average (str): Type of averaging performed on the data ('micro', 'macro', 'weighted', or 'samples').
            multi_class (str): Specifies the multiclass strategy ('ovr', 'ovo', or 'raise' for binary classification).
        """
        super().__init__()
        # self.average = average
        # self.multi_class = multi_class

    def __str__(self):
        return "auroc"
    
    def preprocess_inf(self, x, array):
        if not np.isinf(x):
            return x
        elif x > 0:
            return array.max() + 1
        else:
            return array.min() - 1

    def __call__(self, estimator: List[float], target: List[int]) -> float:
        """
        Measures the AUROC between `estimator` and `target`.

        Parameters:
            estimator (List[float]): A batch of model confidence scores or decision function values.
            target (List[int]): A batch of ground-truth class labels.

        Returns:
            float: Area Under the ROC Curve score.
        """
        target = np.array(target, dtype=np.int32)
        estimator = np.array(estimator, dtype=np.float32)

        estimator = [self.preprocess_inf(x, estimator) for x in estimator]
        # Normalize target values if needed
        target = normalize(target)
        target, estimator = skip_target_nans(target, estimator)

        # Compute AUROC
        # auc_score = roc_auc_score(target, estimator, average=self.average, multi_class=self.multi_class)
        auc_score = roc_auc_score(target, estimator)
        return auc_score
