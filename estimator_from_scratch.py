import numpy as np

from typing import Any, Dict, Literal, Optional
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder


class MultiсlassLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Multiclass Logistic Regression classifier.

    Parameters
    ----------
    max_iter : int, default=1000
        Maximum number of iterations for gradient descent.
    tol : float, default=1e-4
        Tolerance for stopping criteria.
    learning_rate : float, default=0.001
        Step size for gradient descent updates.
    penalty : {'l1', 'l2'}, default=None
        Regularization type. None means no regularization.
    C : float, default=1.0
        Regularization strength (inverse). Ignored if penalty is None.
    random_state : int or None, default=None
        Seed for random number generator to initialize weights.
    """
    def __init__(
        self,
        max_iter: int = 1000,
        tol: float = 1e-4,
        learning_rate: float = 0.001,
        penalty: Optional[Literal["l1", "l2"]] = None,
        C: float = 1.0,
        random_state: Optional[int] = None
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.penalty = penalty
        self.C = C
        self.random_state = random_state

        self.loss_values_ = None
        self.weights_ = None
        self._label_encoder = None

        self._check_invariants()


    def fit(self, X: np.ndarray, y: np.ndarray) -> "MulticlassLogisticRegression":
        """
        Fit the logistic regression model on the training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : np.ndarray of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self : MulticlassLogisticRegression
            The fitted estimator.
        """
        X = np.asarray(X, dtype=float)

        self._label_encoder = LabelEncoder()
        y = self._label_encoder.fit_transform(y)
        y = np.asarray(y, dtype=int)

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Number of samples in X ({X.shape[0]}) does not match number of samples in y ({y.shape[0]})")

        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        X = np.column_stack([np.ones(n_samples), X])
        y = np.eye(n_classes)[y]

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.weights_ = np.random.randn(n_classes, n_features + 1) * 0.01
        self.loss_values_ = []
        loss = np.inf

        for i in range(self.max_iter):
            probabilities = self._predict_proba(X)
            gradient = -np.dot((y - probabilities).T, X) / n_samples

            if self.penalty == 'l1':
                gradient[:, 1:] += 1.0 / self.C * np.sign(self.weights_[:, 1:])
            elif self.penalty == 'l2':
                gradient[:, 1:] += 1.0 / self.C * self.weights_[:, 1:]

            self.weights_ -= self.learning_rate * gradient
            cur_loss = self._loss_function(X, y)
            self.loss_values_.append(cur_loss)

            if np.abs(loss - cur_loss) <= self.tol:
                break

            loss = cur_loss

        return self


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        X = np.column_stack([np.ones(X.shape[0]), X])
        probabilities = self._predict_proba(X)
        y_pred = np.argmax(probabilities, axis=1)
        return self._label_encoder.inverse_transform(y_pred)


    def _check_invariants(self) -> None:
        """
        Validate hyperparameters to ensure they satisfy required conditions.

        Raises
        ------
        ValueError
            If any hyperparameter has an invalid type or value.
        """
        if not isinstance(self.max_iter, int) or self.max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")

        if not isinstance(self.tol, float) or self.tol <= 0:
            raise ValueError("tol must be a positive float")

        if not isinstance(self.learning_rate, float) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be a positive float")

        if self.penalty not in (None, "l1", "l2"):
            raise ValueError("penalty must be None, 'l1', or 'l2'")

        if not isinstance(self.C, float) or self.C <= 0:
            raise ValueError("C must be a positive float")

        if self.random_state is not None:
            if not isinstance(self.random_state, int) or self.random_state < 0:
                raise ValueError("random_state must be a non-negative integer or None")


    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute class probabilities for input samples using softmax.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features + 1)
            Input matrix (including bias term).

        Returns
        -------
        probabilities : np.ndarray of shape (n_samples, n_classes)
            Predicted probabilities for each class.
        """
        return self._softmax(self._logits(X))


    def _logits(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the raw class scores (logits).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features + 1)
            Input matrix (including bias term).

        Returns
        -------
        logits : np.ndarray of shape (n_samples, n_classes)
            Linear combination of inputs and weights.
        """
        return np.dot(X, self.weights_.T)


    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply the softmax function to logits.

        Parameters
        ----------
        logits : np.ndarray of shape (n_samples, n_classes)
            Raw class scores.

        Returns
        -------
        probabilities : np.ndarray of shape (n_samples, n_classes)
            Normalized probabilities for each class.
        """
        exp_Z = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


    def _loss_function(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the regularized cross-entropy loss.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features + 1)
            Input matrix (including bias term).
        y : np.ndarray of shape (n_samples, n_classes)
            One-hot encoded target labels.

        Returns
        -------
        loss : float
            Cross-entropy loss with regularization.
        """
        probabilities = self._predict_proba(X)
        reg = 0

        if self.penalty == 'l1':
            reg = 1.0 / self.C * np.sum(np.abs(self.weights_[:, 1:]))
        elif self.penalty == 'l2':
            reg = 1.0 / self.C * np.sum(self.weights_[:, 1:] ** 2)

        return -np.mean(np.sum(y * np.log(probabilities + 1e-15), axis=1)) + reg