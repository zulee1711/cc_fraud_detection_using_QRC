import numpy as np
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler

class ClassicalReadout:
    """
    Classical linear readout for QUC. The sequence of measured quantum stats for each input sample
    are flattened into a single feature vector and passed to a classical linear classifier.
    """
    def __init__(self, model_type: str = "ridge", alpha: float = 1.0, use_balanced_weights: bool = False, decision_threshold: Optional[float] = None):
        """
        Initialize classical readout

        Args:
            model_type: Ridge Clssifier or Logisitic Regression
            alpha : Regularization parameter for Ridge Classifier. For LogisticRegression, inverse C = 1/alpha is used.
            decision_threshold : Cutoff for binary classification
            use_balanced_weights : Only used if decision_threshold is None. Applies balanced loss weights
        """
        self.model_type = model_type.lower()
        self.alpha = alpha
        self.scaler = StandardScaler()  # normalizes each feature using statistics calculated from training set              

        # Set threshold manually
        if decision_threshold is not None:
             self.class_weight = None
             self.decision_threshold = decision_threshold
        else :
             self.class_weight = "balanced" if use_balanced_weights else None
             self.decision_threshold = 0.0 if self.model_type == "ridge" else 0.5

        # Select classical model for readout
        if model_type == "ridge":
            self.model = RidgeClassifier(alpha=alpha, class_weight=self.class_weight)
        elif model_type == "logistic":
            self.model = LogisticRegression(C=1.0/alpha, class_weight=self.class_weight, max_iter=1000)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def prepare_features(self, reservoir_outputs: np.ndarray, fit_scaler: bool = False) -> np.ndarray:
        """
        Flattens reservoir matrices from (Batch Size M, Sequence Length L, Qubits N)
        into a flat feature vector (M, L * N).

        Args :
            reservoir_outputs : reservoir states with shape (M, L, N)
            fit_scaler : whether to fit the StandardScaler before transforming
        """
        # Number of independent input samples
        num_samples = reservoir_outputs.shape[0]
        # Reshape
        X_flat = reservoir_outputs.reshape(num_samples, -1)
        # Fitting scaler only when processing training data
        if fit_scaler:
            return self.scaler.fit_transform(X_flat)
        return self.scaler.transform(X_flat)

    def _get_scores(self, X: np.ndarray) -> np.ndarray:
         if self.model_type == "logistic":
              return self.model.predict_proba(X)[:,1]   # prob in [0,1] that a sequence is fraud
         elif self.model_type == "ridge":
              return self.model.decision_function(X)    # distance to hyperplane (-inf, +inf)
         
    
    def fit_evaluate(
            self, 
            reservoir_outputs: np.ndarray, 
            y_labels: np.ndarray, 
            test_size: float = 0.3, 
            random_state: int = 42
        ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
            """
            Flattens reservoir states, splits data, trains readout classifier, 
            and computes performance metrics.
            """

            # Split data into training and testing subsets
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(reservoir_outputs, y_labels, test_size=test_size, random_state=random_state, stratify=y_labels)

            # Convert training and test data into flat feature vectors
            X_train = self.prepare_features(X_train_raw, fit_scaler=True)
            X_test = self.prepare_features(X_test_raw, fit_scaler=False)

            self.model.fit(X_train, y_train)

            # Extract decision scores
            test_scores = self._get_scores(X_test)

            # Apply thresholds
            y_pred = (test_scores >= self.decision_threshold).astype(int)

            # Calculate metrics
            metrics = {
                "f1_score": f1_score(y_test, y_pred, zero_division=0),
                "threshold_used": self.decision_threshold
            }

            return metrics, y_test, y_pred
