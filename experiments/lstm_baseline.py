"""Classical LSTM baseline for the fraud-detection sequence experiment."""

from __future__ import annotations

import copy
import time

import numpy as np
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from qrc.decision import select_threshold_min_fp


def run_lstm_baseline(
    splits,
    *,
    seed=42,
    hidden_size=32,
    epochs=50,
    batch_size=256,
    learning_rate=1e-3,
    patience=8,
    min_recall=0.8,
):
    """Fit an LSTM on training windows and score the shared test split.

    The scaler is fit only on training windows. Validation loss selects the
    best epoch; validation and test metrics are then reported.
    """
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError(
            "The LSTM baseline requires PyTorch. Install it with "
            '`pip install -e ".[classical]"`.'
        ) from exc
    from qrc.baselines.lstm import LSTMClassifier

    X_train, y_train = splits["train"]
    X_validation, y_validation = splits["validation"]
    X_test, y_test = splits["test"]
    X_train = np.asarray(X_train, dtype=np.float32)
    X_validation = np.asarray(X_validation, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)
    y_validation = np.asarray(y_validation, dtype=np.float32).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)

    if X_train.ndim != 3:
        raise ValueError(f"LSTM inputs must have shape (samples, timesteps, features), got {X_train.shape}")
    if any(len(X) != len(y) for X, y in (
        (X_train, y_train),
        (X_validation, y_validation),
        (X_test, y_test),
    )):
        raise ValueError("each split's inputs and labels must have the same number of samples")
    if not len(X_train) or not len(X_validation) or not len(X_test):
        raise ValueError("train, validation, and test splits must all be non-empty")
    if epochs < 1 or batch_size < 1 or hidden_size < 1 or patience < 1:
        raise ValueError("epochs, batch_size, hidden_size, and patience must be positive")

    scaler = StandardScaler()
    feature_count = X_train.shape[2]
    scaler.fit(X_train.reshape(-1, feature_count))

    def scale(windows):
        shape = windows.shape
        return scaler.transform(windows.reshape(-1, feature_count)).reshape(shape).astype(np.float32)

    X_train = scale(X_train)
    X_validation = scale(X_validation)
    X_test = scale(X_test)

    # fix segfault on macOS (torch and qiskit-aer each bundle their own libomp)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    model = LSTMClassifier(input_size=feature_count, hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.BCEWithLogitsLoss()
    train_inputs = torch.from_numpy(X_train)
    train_targets = torch.from_numpy(y_train)
    validation_inputs = torch.from_numpy(X_validation)
    validation_targets = torch.from_numpy(y_validation)

    start = time.perf_counter()
    best_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(train_inputs), generator=generator)
        for indices in order.split(batch_size):
            logits = model(train_inputs[indices])
            loss = loss_function(logits, train_targets[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_loss = loss_function(
                model(validation_inputs), validation_targets
            ).item()
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    elapsed = time.perf_counter() - start
    model.eval()

    # Score validation first, tune the operating threshold on validation only,
    # then freeze it for both validation and test reporting.
    with torch.no_grad():
        validation_scores = torch.sigmoid(model(torch.from_numpy(X_validation))).numpy()
        test_scores = torch.sigmoid(model(torch.from_numpy(X_test))).numpy()

    decision_threshold = select_threshold_min_fp(
        y_validation, validation_scores, min_recall=min_recall
    )

    metrics, scores = {}, {}
    for split, labels, split_scores in (
            ("validation", y_validation, validation_scores),
            ("test", y_test, test_scores),
    ):
        predictions = (split_scores >= decision_threshold).astype(int)
        negative = labels == 0
        false_positive_rate = (
            float(np.mean(predictions[negative] == 1)) if np.any(negative) else 0.0
        )
        metrics[split] = {
            "average_precision": average_precision_score(labels, split_scores),
            "roc_auc": roc_auc_score(labels, split_scores),
            "f1_score": f1_score(labels, predictions, zero_division=0),
            "precision": precision_score(labels, predictions, zero_division=0),
            "recall": recall_score(labels, predictions, zero_division=0),
            "false_positive_rate": false_positive_rate,
            "matthews_corr_coef": matthews_corrcoef(labels, predictions),
            "balanced_accuracy": balanced_accuracy_score(labels, predictions),
            "threshold_used": decision_threshold,
        }
        scores[split] = split_scores

    return {"backend": "lstm", "seconds": elapsed, "metrics": metrics, "scores": scores}
