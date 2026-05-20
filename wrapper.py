import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class KerasClassifierWrapper(BaseEstimator, ClassifierMixin):
    """sklearn-compatible wrapper around a pre-trained Keras model.

    Defined in a real module (not a notebook namespace) so that the
    StackingClassifier containing it can be pickled and reloaded by
    the GUI without `__main__` lookup failures.
    """

    def __init__(self, model=None, num_classes=5):
        self.model = model
        self.num_classes = num_classes
        self.classes_ = np.arange(num_classes)

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.argmax(self.model.predict(X, verbose=0), axis=1)

    def predict_proba(self, X):
        return self.model.predict(X, verbose=0)
