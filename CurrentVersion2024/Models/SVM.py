from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
import numpy as np
import pickle
import os

class SVM:
    """A simple interface between a KMeans model and the generic ML-QuIC structure. All the methods are unaltered, just a mask to keep things consistent."""

    def __init__(self, kernel = 'rbf', degree = 3, random_state = 7, file_path = './', model_name = 'svm'):
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
            self.model = SVC(kernel=kernel, degree = degree, random_state=random_state)
            self.scaler = StandardScaler()
            self.pca = PCA(n_components=4)

    def fit(self, x = None, y = None):
        if not self.pretrained:
            x = self.pca.fit_transform(x)
            self.scaler.fit(x)
            x = self.scaler.transform(x)
            y = np.array(y == 2)
            self.model.fit(x, y)
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
                
            with open(self.pca_path, 'wb') as f:
                pickle.dump(self.pca, f)
        
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
    
    def predict(self, data, binary = True):
        """Binary is unimplemented because SVM is a true binary classifier with no ambiguity"""
        data = self.pca.transform(data)
        data = self.scaler.transform(data)
        return self.model.predict(data)
    
    def get_scores(self, data, true):
        pred = self.predict(data)
        true = np.where(true == 2, 1, 0)
        return classification_report(true, pred, target_names=["neg", "pos"])