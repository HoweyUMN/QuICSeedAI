from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
import numpy as np

class KMeansModel:
    """A simple interface between a KMeans model and the generic ML-QuIC structure. All the methods are unaltered, just a mask to keep things consistent."""

    def __init__(self, n_clusters = 2, random_state = 7):
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.pca = PCA(n_components=2)

    def fit(self, x = None, y = None):
        x = self.pca.fit_transform(x)
        self.model.fit(x)

    
    def predict(self, data, binary = True):
        """Binary is unimplemented because KMeans returns true binary output"""
        data = self.pca.transform(data)
        return self.model.predict(data)
    
    def get_scores(self, data, true):
        pred = self.predict(data, binary=True)
        if np.max(pred) == 2: 
            pred = np.where(pred == 2, 1, 0)
        true = np.where(true == 2, 1, 0)

        # KMeans picks whatever label it wants, so we assume if its doing under 50, it picked opposite to us
        if len(pred[pred == true]) < 0.5 * len(true):
            pred = 1 - pred

        return classification_report(true, pred, target_names=["neg", "pos"])