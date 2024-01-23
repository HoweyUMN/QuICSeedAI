from sklearn.cluster import SpectralClustering as SC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
import numpy as np

class SpectralClustering:

    def __init__(self, n_clusters = 2, n_vars = 2):
        """Creates the model of a specified type to be the underlying model"""
        self.model = SC(n_clusters=n_clusters, affinity='nearest_neighbors', assign_labels='discretize', n_jobs=12)
        self.labels = []
        self.learned_data = np.zeros(1)
        self.pca = PCA(n_components=min(n_vars, 8))

    def fit(self, x = None, y = None):
        """Acts as an interface for the underlying model's fit method, converting standardized data
        into a format which the MLP can recognize"""
        # x = self.pca.fit_transform(x)
        pass

    def predict(self, data, binary = False):
        """Acts as an interface for the model's prediction method, performs scaling and
        gets data into a format appropriate for testing"""

        # Save time because this takes forever
        if np.array_equal(self.learned_data, data):
            return self.labels

        # Get the predictions
        data = self.pca.fit_transform(data)
        return self.model.fit_predict(data)

    def get_scores(self, data, true):
        """Acts as an interface to get a model's predictions and obtain metrics from them"""
        pred = self.predict(data, binary=True)
        if np.max(pred) == 2: 
            pred = np.where(pred == 2, 1, 0)
        true = np.where(true == 2, 1, 0)

        # Spectral Clustering picks whatever label it wants, so we assume if its doing under 50, it picked opposite to us
        if len(pred[pred == true]) < 0.5 * len(true):
            pred = 1 - pred

        return classification_report(true, pred, target_names=["neg", "pos"])
