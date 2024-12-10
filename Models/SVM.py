from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
import numpy as np
import pickle
import os

class SVM:
    """A simple interface between a KMeans model and the generic ML-QuIC structure. All the methods are unaltered, just a mask to keep things consistent."""

    def __init__(self, kernel = 'rbf', random_state = 333, file_path = './', model_name = 'svm', pca = True):
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
            self.model = SVC(kernel=kernel, random_state=random_state, gamma = 'auto', probability=True)
            self.scaler = StandardScaler()
            self.pca_model = PCA(n_components=4)
        
        print('\nSVM Model Loaded:')
        print(type(self.model))

    def fit(self, x = None, y = None):
        if not self.pretrained:
            self.scaler.fit(x)
            x = self.scaler.transform(x)
            if self.pca:
                x = self.pca_model.fit_transform(x)
            
            y = np.array(y == 2)
            self.model.fit(x, y)
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
                
            with open(self.pca_path, 'wb') as f:
                pickle.dump(self.pca_model, f)
        
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
    
    def predict(self, data, labels = None, binary = True):
        """Binary is unimplemented because SVM is a true binary classifier with no ambiguity
        - Labels are unused"""
        data = StandardScaler().fit_transform(data)
        if self.pca:
            data = self.pca_model.transform(data)
        
        return self.model.predict(data)
    
    def get_scores(self, data, true):
        pred = self.predict(data)
        true = np.where(true == 2, 1, 0)
        return classification_report(true, pred, target_names=["neg", "pos"])