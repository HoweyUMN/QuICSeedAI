from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
import numpy as np
import pickle

class SVM:
    """A simple interface between a KMeans model and the generic ML-QuIC structure. All the methods are unaltered, just a mask to keep things consistent."""

    def __init__(self, kernel = 'rbf', degree = 3, random_state = 7, file_path = None):
        if file_path is None:
            self.model = SVC(kernel=kernel, degree = degree, random_state=random_state)
            self.scaler = StandardScaler()
            self.pca = PCA(n_components=4)
            self.file_path = 'NULL'
        else:
            self.model = pickle.load(open(file_path + 'svm.pkl', 'rb'))
            self.scaler = pickle.load(open(file_path + 'svm_scaler.pkl', 'rb'))
            self.pca = pickle.load(open(file_path + 'svm_pca.pkl', 'rb'))
            self.file_path = file_path

    def fit(self, x = None, y = None):
        if self.file_path == 'NULL':
            x = self.pca.fit_transform(x)
            self.scaler.fit(x)
            x = self.scaler.transform(x)
            y = np.array(y == 2)
            self.model.fit(x, y)
            
            with open('../svm.pkl', 'wb') as f:
                pickle.dump(self.model, f)
                
            with open('../svm_pca.pkl', 'wb') as f:
                pickle.dump(self.pca, f)
        
            with open('../svm_scaler.pkl', 'wb') as f:
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