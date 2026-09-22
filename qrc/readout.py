""""
The quantum reservoir produces one final measurement vector for each input sequence.
For a reservoir with n qubits, the readout contains:
    - n local Z expectation values : <Z_i>
    - n-1 nearest-neighbor ZZ expectation values : <Z_i Z_{i+1}>

The number of measured observables :
    N = 2*n - 1
    
    where : n = n_in + n_mem

For M input sequences, the reservoir has shape :
    (M, N)

The classical readout standardizees the quantum measurements and uses 
logistic regression to predict whether each sequence correspond to fraud or not.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple
from sklearn.preprocessing import StandardScaler

class ClassicalReadout:
    """
    Logistic regression classifier for quantum reservoir outputs.
    """
    def __init__(self, alpha: float = 1.0, use_balanced_weights: bool = False, decision_threshold: float = 0.5):
        """
        Initialize classical readout

        Args:
            alpha : Regularization parameter. For LogisticRegression, inverse C = 1/alpha is used.
            decision_threshold : Probability threshold to convert predicted fraud probs into binary predictions
            use_balanced_weights : If True, compensates for class imbalance during training.
        """
        self.alpha = alpha
        self.decision_threshold = decision_threshold
        self.class_weight = "balanced" if use_balanced_weights else None
        self.scaler = StandardScaler()  # normalizes each feature using statistics calculated from training set              

        self.model = LogisticRegression(C=1.0/alpha, class_weight=self.class_weight, max_iter=1000)

    def prepare_features(self, reservoir_outputs: np.ndarray, fit_scaler: bool = False) -> np.ndarray:
        """
        Scale reservoir measurements.
        Expected shape : 
            (M, N)

            where : 
                M = number of input sequences
                N = number of measured obesrvables

        Args :
            reservoir_outputs : final quantum measurements for all input sequences. Expected shape : (M, N)
            fit_scaler : whether to fit the StandardScaler before transforming
        
        Returns : 
            The standardized resrevoir measurements with the same shape (M, N) as the input.
        """
        if fit_scaler:
            return self.scaler.fit_transform(reservoir_outputs)
        return self.scaler.transform(reservoir_outputs)


    def fit_evaluate(
            self, 
            reservoir_outputs: np.ndarray, 
            y_labels: np.ndarray, 
            test_size: float = 0.3, 
            random_state: int = 42
        ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
            """
            Splits data, trains readout classifier, 
            and computes performance metrics.

            Args :
                reservoir_outputs : final quantum measurements
                y_labels : Binary target labels for the M input sequences (M,)
                test_size : Fraction of the input sequences reserved for the test set.
                random_state : Random seed used to make the train/test split reproducible.

            Returns : 
                A tuple containing :
                    metrics : Dictionary containing F1 score, precision, recall, ROC-AUC and prob threshold used.
                    y_test : True labels for the test sequences (M_test,)
                    y_pred : Binary predictions produced using decision_threshold (M_test,)
                    test_scores : Predicted probability of fraud for each test sequences (M_test,)
            """

            # Split data into training and testing subsets
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(reservoir_outputs, y_labels, test_size=test_size, random_state=random_state, stratify=y_labels)

            # Standardize training and test reservoir measurements
            X_train = self.prepare_features(X_train_raw, fit_scaler=True)
            X_test = self.prepare_features(X_test_raw, fit_scaler=False)

            self.model.fit(X_train, y_train)

            # Predict probability that each test sequence is fraudulent
            test_scores = self.model.predict_proba(X_test)[:, 1]

            # Apply thresholds
            y_pred = (test_scores >= self.decision_threshold).astype(int)

            # Calculate metrics
            metrics = {
                "f1_score": f1_score(y_test, y_pred, zero_division=0),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, test_scores),
                "threshold_used": self.decision_threshold
            }

            return metrics, y_test, y_pred, test_scores
