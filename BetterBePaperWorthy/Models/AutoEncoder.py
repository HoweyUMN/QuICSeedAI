import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model, regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

class AutoEncoder:
    """A handy interface for ML-QuIC to interact with the Autoencoder format"""

    def __init__(self, NDIM):
        input_layer = Input(shape=NDIM)
        ael = Dense(NDIM / 2, activation = 'relu', kernel_regularizer = regularizers.L1L2(1e-4, 1e-4))(input_layer)
        ael = Dense(NDIM / 4, activation = 'relu', kernel_regularizer = regularizers.L1L2(1e-4, 1e-4))(ael)
        ael = Dense(NDIM / 8, activation = 'relu', kernel_regularizer = regularizers.L1L2(1e-4, 1e-4))(ael)
        ael = Dense(NDIM / 16, activation = 'relu', kernel_regularizer = regularizers.L1L2(1e-4, 1e-4))(ael)

        ael = Dense(NDIM / 8, activation = 'relu', kernel_regularizer = regularizers.L1L2(1e-4, 1e-4))(ael)
        ael = Dense(NDIM / 4, activation = 'relu', kernel_regularizer = regularizers.L1L2(1e-4, 1e-4))(ael)
        ael = Dense(NDIM / 2, activation = 'relu', kernel_regularizer = regularizers.L1L2(1e-4, 1e-4))(ael)
        output = Dense(NDIM, activation='tanh')(ael)

        self.model = Model(input_layer, output)
        self.scaler = StandardScaler()

    def fit(self, learning_rate=1e-4, loss = 'mse',
            x = None, y = None, batch_size = 128, epochs = 500, verbose = 0, callbacks = None, validation_split = 0.1,
            validation_data = None, shuffle = True, class_weight = None, sample_weight=None, initial_epoch=0, 
            steps_per_epoch = None, validation_steps = None, validation_batch_size = None, validation_freq = 1, 
            max_queue_size = 10, workers = 1, use_multiprocessing = False):
        
        self.scaler.fit(x)
        x = self.scaler.transform(x)

        self.model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate), loss = loss)

        self.model.fit(
            x,
            x,
            batch_size,
            epochs,
            verbose,
            callbacks,
            validation_split,
            validation_data,
            shuffle,
            class_weight,
            sample_weight,
            initial_epoch,
            steps_per_epoch,
            validation_steps,
            validation_batch_size,
            validation_freq,
            max_queue_size,
            workers,
            use_multiprocessing,
        )

    def predict(self, data):
	data = self.scaler.transform(data)
        return self.model.predict(data)
    
    def get_scores(self, data, true):
        preds = self.model.predict(data)
        mses = np.zeros(len(preds))
        for i,pred in enumerate(preds):
            errors = pred - data[i]
            sq_err = np.zeros(len(errors))
            for j,err in enumerate(errors):
                sq_err[j] = err**2
            mses[i] = np.mean(sq_err)
        
        score_preds = np.where(mses > np.mean(mses), 1, 0)
        return classification_report(true, score_preds, target_names=['neg', 'pos'])
            


