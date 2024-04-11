from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import os

class KMeansModel:
    """A simple interface between a KMeans model and the generic ML-QuIC structure. All the methods are unaltered, just a mask to keep things consistent."""

    def __init__(self, n_clusters = 2, random_state = 7, file_path = './', model_name = 'kmeans'):
        self.model_path = file_path + model_name + '.pkl'
        self.scaler_path = file_path + model_name + '_scaler.pkl'
        self.pca_path = file_path + model_name + '_pca.pkl'
        self.pretrained = False
        
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path) and os.path.exists(self.pca_path):
            self.model = pickle.load(open(self.model_path, 'rb'))
            self.scaler = pickle.load(open(self.scaler_path, 'rb'))
            self.pca = pickle.load(open(self.pca_path, 'rb'))
            self.pretrained = True
        
        else: # Generate new if doesn't exist
            self.model = KMeans(n_clusters=n_clusters, random_state=random_state)
            self.pca = PCA(n_components=4)
            self.scaler = StandardScaler()

    def fit(self, x = None, y = None):
        if not self.pretrained:
            x = self.pca.fit_transform(x)
            x = self.scaler.fit_transform(x)
            
            self.model.fit(x)

    
    def predict(self, data, binary = True):
        """Binary is unimplemented because KMeans returns true binary output"""
        data = self.pca.transform(data)
        data = self.scaler.transform(data)
        preds = self.model.predict(data)
        max = np.max(preds)
        preds = np.array(preds == max, dtype=int) # Fix an issue where KMeans picks weird labels

        return preds
    
    def get_scores(self, data, true):
        pred = self.predict(data, binary=True)
        if np.max(pred) == 2: 
            pred = np.where(pred == 2, 1, 0)
        true = np.where(true == 2, 1, 0)

        # KMeans picks whatever label it wants, so we assume if its doing under 50, it picked opposite to us
        if len(pred[pred == true]) < 0.5 * len(true):
            pred = 1 - pred

        return classification_report(true, pred, target_names=["neg", "pos"])