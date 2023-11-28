from sklearn.cluster import KMeans
from sklearn.metrics import classification_report

class KMeansModel:
    """A simple interface between a KMeans model and the generic ML-QuIC structure. All the methods are unaltered, just a mask to keep things consistent."""

    def __init__(self, n_clusters = 2, random_state = 7):
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state)

    def fit(self, x = None, y = None):
        self.model.fit(x)
    
    def predict(self, data):
        return self.model.predict(data)
    
    def get_scores(self, data, true):
        pred = self.predict(data)
        return classification_report(true, pred, target_names=["neg", "pos"])