from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np

class SVM:
    """A simple interface between a KMeans model and the generic ML-QuIC structure. All the methods are unaltered, just a mask to keep things consistent."""

    def __init__(self, kernel = 'rbf', degree = 3, random_state = 7, ):
        self.model = SVC(kernel=kernel, degree = degree, random_state=random_state)
        self.scaler = StandardScaler()

    def fit(self, x = None, y = None):
        self.scaler.fit(x)
        x = self.scaler.transform(x)
        self.model.fit(x, y)
    
    def predict(self, data):
        data = self.scaler.transform(data)
        return self.model.predict(data)
    
    def get_scores(self, data, true):
        pred = self.predict(data)
        true = np.where(true >= 2, 1, 0)
        return classification_report(true, pred, target_names=["neg", "pos"])