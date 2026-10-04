""""
The quantum reservoir produces one final measurement vector for each input sequence.
Which observables are measured is specified earlier in the protocol, and it defines the number of measured observables N.
For example, for a reservoir with n qubits, and a choice of measuring all n local Z expectation values (the <Z_i>)
and all n-1 nearest-neighbor ZZ expectation values (the <Z_i Z_{i+1}>), we have N = 2*n - 1.

For M input sequences, the reservoir has shape :
    (M, N)

This module provides two classical readouts :

- ClassicalReadout : logistic regression for binary classifcation
- RegressionReadout : ridge regression for continuous-target tasks

Both readouts standardize the reservoir measurements using statistics fitted
only on the training data. The same fitted scaler is then reused unchanged
for validation and test data.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
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
                N = number of measured observables

        Args :
            reservoir_outputs : final quantum measurements for all input sequences. Expected shape : (M, N)
            fit_scaler : whether to fit the StandardScaler before transforming
        
        Returns : 
            The standardized resrevoir measurements with the same shape (M, N) as the input.
        """
        if fit_scaler:
            return self.scaler.fit_transform(reservoir_outputs)
        return self.scaler.transform(reservoir_outputs)


    def fit(self, train_outputs: np.ndarray, y_train: np.ndarray,) -> None :
         """
         Fit the scaler and logistic regression model on training data.

         Args :
            train_outputs : Quantum resrvoir measurements for the training sequences, with shape (M_train, N)
            y_train : Binary training labels, with shape (M_train, )
         """
         # Fit the scaler using training data
         X_train = self.prepare_features(train_outputs, fit_scaler=True)

         # Fit logistic regression model using standardized training data
         self.model.fit(X_train, y_train)


    def predict(self, reservoir_outputs: np.ndarray) -> np.ndarray : 
         """
         Predict fraud probs using the fitted readout.

         Args : 
            reservoir_outputs : Quantum reservoir measurements to classify, with shape (M, N)

         Returns :
            Fraud probabilities with shape (M, )
         """
         X = self.prepare_features(reservoir_outputs, fit_scaler=False)
         return self.model.predict_proba(X)[:,1]

    def evaluate(self, reservoir_outputs: np.ndarray, y_labels: np.ndarray) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate an already fitted classifier on the dataset

        Args :
            reservoir_outputs : Quantum reservoir measurements for the sequences to evaluate with shape (M, N)
            y_labels : True binary labels for the sequences, with shape (M, )

        Returns:
            A tuple containing:
            metrics: Dictionary containing F1 score, precision, recall, ROC-AUC, PR-AUC and the decision threshold used.
            y_labels: The true labels
            y_pred: Binary predictions obtained using decision_threshold
            scores: Predicted probability of fraud for each sequence

        """
        # Predict fraud probs using already fitted model
        scores = self.predict(reservoir_outputs)

        # Convert probs into binary predictions
        y_pred = (scores >= self.decision_threshold).astype(int)

        # Calculate matrics
        metrics = {
            "f1_score": f1_score(
                y_labels,
                y_pred,
                zero_division=0,
            ),
            "precision": precision_score(
                y_labels,
                y_pred,
                zero_division=0,
            ),
            "recall": recall_score(
                y_labels,
                y_pred,
                zero_division=0,
            ),
            "roc_auc": roc_auc_score(
                y_labels,
                scores,
            ),
            "pr_auc": average_precision_score(
                y_labels,
                scores,
            ),
            "threshold_used": self.decision_threshold,
        }
        return metrics, y_labels, y_pred, scores



class RegressionReadout:  
    """
    Ridge regression readout for continuous targets (for benchmarks such as the memory task).

    The train/test split is left to the
    caller: benchmark windows overlap in time, so the split must be chronological.
    """
    def __init__(self, alpha: float = 1e-6):
        """
        Args:
            alpha : Ridge regularization strength. Kept small by default: the
                benchmarks are noise-free, so we want to see what the reservoir
                features can express.
        """
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)


    def prepare_features(self, reservoir_outputs: np.ndarray, fit_scaler: bool = False) -> np.ndarray:
            """
            Scale reservoir measurements.
            Expected shape : 
                (M, N)
    
                where : 
                    M = number of input sequences
                    N = number of measured observables
    
            Args :
                reservoir_outputs : final quantum measurements for all input sequences. Expected shape : (M, N)
                fit_scaler : whether to fit the StandardScaler before transforming
            
            Returns : 
                The standardized resrevoir measurements with the same shape (M, N) as the input.
            """
            if fit_scaler:
                return self.scaler.fit_transform(reservoir_outputs)
            return self.scaler.transform(reservoir_outputs)


    def fit(self, train_outputs: np.ndarray, y_train: np.ndarray) -> None :
         """
         Fit the scaler and ridge regression model on training data.

        Args:
            train_outputs: Quantum reservoir measurements for the training sequences, with shape (M_train, N).
            y_train: Continuous training targets, with shape (M_train,)
         
         """
         # Fit scaler using training data
         X_train = self.prepare_features(train_outputs, fit_scaler = True)

         # Fit ridge regression model using standardized training data
         self.model.fit(X_train, y_train)

    def predict(self, reservoir_outputs: np.ndarray,) -> np.ndarray:
         """
         Predicts continuous target values using the fitted readout
         """
         X = self.prepare_features(reservoir_outputs, fit_scaler=False)
         return self.model.predict(X)

    def evaluate(self, reservoir_outputs: np.ndarray, y_labels: np.ndarray,) -> Tuple[Dict[str, float], np.ndarray]:
         # Predict using already fitted model
         y_pred = self.predict(reservoir_outputs)

         # Calculate normalized mean squared error
         nmse = np.mean((y_pred - y_labels)**2) / np.var(y_labels)

         # A constant prediction has no correlation with anything - avoid NaN in that case
         capacity = (0.0 if np.std(y_pred) == 0 else np.corrcoef(y_pred, y_labels)[0,1]**2)

         metrics = {
            "nmse": float(nmse),
            "capacity": float(capacity),
        }
         return metrics, y_pred


"""
INTENDED USE :

Each X split must be first be passed through the quantum reservoir to ontain the
reservoir measurement vectors:

reservoir_train = protocol.run(X_train) 
reservoir_validation = protocol.run(X_validation) 
reservoir_test = protocol.run(X_test)


The readout is then fitted only on the training reservoir outputs, as follows : 


CLASSIFICATION

readout = ClassicalReadout(alpha=1.0)
readout.fit(reservoir_train, y_train)
validation_metrics, y_val, y_val_pred, val_scores = readout.evaluate(reservoir_validation, y_validation)
test_metrics, y_test, y_test_pred, test_scores = readout.evaluate(reservoir_test, y_test)

REGRESSION 

readout = RegressionReadout(alpha=1e-6)
readout.fit(reservoir_train, y_train)
validation_metrics, validation_predictions = readout.evaluate(reservoir_validation, y_validation)
test_metrics, test_predictions = readout.evaluate(reservoir_test, y_test)
"""
