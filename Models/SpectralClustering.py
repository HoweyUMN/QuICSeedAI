from sklearn.cluster import SpectralClustering as SC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, f1_score
import numpy as np

class SpectralClustering:

    def __init__(self, n_clusters = 2, n_vars = 4):
        """Creates the model of a specified type to be the underlying model"""
        self.model = SC(n_clusters=n_clusters, affinity='nearest_neighbors', assign_labels='discretize', n_jobs=12)
        self.labels = None
        self.learned_data = np.zeros(1)
        # self.pca = PCA(n_components=min(n_vars, 8))

    def fit(self, x = None, y = None):
        """Acts as an interface for the underlying model's fit method, converting standardized data
        into a format which the MLP can recognize"""
        # x = self.pca.fit_transform(x)
        pass

    def predict(self, data, labels, binary = False):
        """Acts as an interface for the model's prediction method, performs scaling and
        gets data into a format appropriate for testing - labels are used to ensure cluster
        labels line up with true labels"""

        # Save time because this takes forever
        if np.array_equal(self.learned_data, data):
            return self.labels

        # Get the predictions
        preds = self.model.fit_predict(data)
        pos_label = np.max(preds) if not binary else 1
        
        if pos_label == 1:
            labels = np.array(labels == 2, dtype=int)
            if np.max(preds) > 1:
                preds = np.array(preds == 2, dtype=int)
        
        f1 =  f1_score(labels, preds, pos_label=pos_label)
        f1_backwards = f1_score(labels, pos_label - preds, pos_label=pos_label)
        
        self.labels = preds if f1 > f1_backwards else pos_label - preds
        self.learned_data = data
        
        return self.labels

    def get_scores(self, data, true):
        """Acts as an interface to get a model's predictions and obtain metrics from them"""
        pred = self.predict(data, true, binary=True)
        if np.max(pred) == 2: 
            pred = np.where(pred == 2, 1, 0)
        true = np.where(true == 2, 1, 0)

        return classification_report(true, pred, target_names=["neg", "pos"])
