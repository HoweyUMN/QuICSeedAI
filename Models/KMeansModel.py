from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np
import pickle
import os

class KMeansModel:
    """A simple interface between a KMeans model and the generic ML-QuIC structure. All the methods are unaltered, just a mask to keep things consistent."""

    def __init__(self, n_clusters = 2, random_state = 7, file_path = './', model_name = 'kmeans', pca = True):
        self.model_path = file_path + model_name + '.pkl'
        self.scaler_path = file_path + model_name + '_scaler.pkl'
        self.pca_path = file_path + model_name + '_pca.pkl'
        self.pretrained = False
        self.pca = pca
        
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path) and os.path.exists(self.pca_path):
            self.model = pickle.load(open(self.model_path, 'rb'))
            self.scaler = pickle.load(open(self.scaler_path, 'rb'))
            self.pca_model = pickle.load(open(self.pca_path, 'rb'))
            self.pretrained = True
        
        else: # Generate new if doesn't exist
            self.model = KMeans(n_clusters=n_clusters, init='random', max_iter=500, n_init=200, random_state=random_state)
            self.pca_model = PCA(n_components=4)
            self.scaler = MinMaxScaler([0, 1])
            
        print('\nKMeans Model Loaded:')
        print(type(self.model))

    def fit(self, x = None, y = None):  
        if not self.pretrained:
            x = self.scaler.fit_transform(x)
            if self.pca:
                x = self.pca_model.fit_transform(x)
            
            self.model.fit(x)
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
                
            with open(self.pca_path, 'wb') as f:
                pickle.dump(self.pca_model, f)
        
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)

    
    def predict(self, data, labels, binary = True):
        """Makes predictions about data"""

        data = MinMaxScaler([0, 1]).fit_transform(data)
        if self.pca:
            data = self.pca_model.fit_transform(data)
        
        preds = self.model.predict(data)
        max = np.max(preds)
        preds = np.array(preds == max, dtype=int) # Fix an issue where KMeans picks weird labels
        
        pos_label = np.max(preds) if not binary else 1
        if pos_label == 1:
            labels = np.array(labels == 2, dtype=int)
            if np.max(preds) > 1:
                preds = np.array(preds == 2, dtype=int)
        
        f1 =  f1_score(labels, preds, pos_label=pos_label)
        f1_backwards = f1_score(labels, pos_label - preds, pos_label=pos_label)
        
        preds = preds if f1 > f1_backwards else pos_label - preds

        return preds
    
    def get_scores(self, data, true):
        pred = self.predict(data, true, binary=True)
        if np.max(pred) == 2: 
            pred = np.where(pred == 2, 1, 0)
        true = np.where(true == 2, 1, 0)

        return classification_report(true, pred, target_names=["neg", "pos"])