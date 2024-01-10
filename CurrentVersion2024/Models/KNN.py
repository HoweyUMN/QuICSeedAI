from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import numpy as np

class KNN:

    def __init__(self, NDIM):
        """Creates the model of a specified type to be the underlying model"""
        self.model = None

    def fit(self, x = None, y = None):
        """Acts as an interface for the underlying model's fit method, converting standardized data
        into a format which the MLP can recognize"""
        self.model.fit(
            x,
            y,
        )

    def predict(self, data, binary = False):
        """Acts as an interface for the model's prediction method, performs scaling and
        gets data into a format appropriate for testing"""

        # Get the predictions
        preds = self.model.predict(data)
        return preds

    def get_scores(self, data, true):
        """Acts as an interface to get a model's predictions and obtain metrics from them"""
        pred = self.predict(data, binary=True)
        true = np.where(true > 1, 1, 0)

        class_report = classification_report(true, pred, target_names=["neg", "pos"])
        return class_report
