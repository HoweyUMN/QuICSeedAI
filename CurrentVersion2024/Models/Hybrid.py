import numpy as np
import tensorflow as tf
from keras.layers import Dense, Input
from keras import Model, regularizers
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class Hybrid:
    """A supervised learning approach implementing an MLP/Autoencoder combination"""

    def __init__(self, NDIM):
        """Creates an Autoencoder and MLP which can be trained together for better accuracy"""

        # Define an MLP
        input_layer = Input(shape=2 * NDIM)
        dense = Dense(NDIM, activation = 'relu', kernel_regularizer = regularizers.L1L2(1e-4, 1e-4))(input_layer)
        dense = Dense(NDIM, activation = 'relu', kernel_regularizer = regularizers.L1L2(1e-4, 1e-4))(dense)
        output = Dense(1, activation='sigmoid')(dense)

        self.mlp = Model(input_layer, output)

        # Define an Autoencoder
        input_layer = Input(shape=NDIM)
        ael = Dense(NDIM / 2, activation = 'relu', kernel_regularizer = regularizers.L1L2(1e-4, 1e-4))(input_layer)
        ael = Dense(NDIM / 4, activation = 'relu', kernel_regularizer = regularizers.L1L2(1e-4, 1e-4))(ael)
        ael = Dense(NDIM / 8, activation = 'relu', kernel_regularizer = regularizers.L1L2(1e-4, 1e-4))(ael)
        ael = Dense(NDIM / 4, activation = 'relu', kernel_regularizer = regularizers.L1L2(1e-4, 1e-4))(ael)
        ael = Dense(NDIM / 2, activation = 'relu', kernel_regularizer = regularizers.L1L2(1e-4, 1e-4))(ael)
        output = Dense(NDIM, activation='tanh')(ael)

        self.ae = Model(input_layer, output)

        # Create a standard scaler for the data
        self.scaler = StandardScaler()


    def fit(self, x = None, y = None):
        """Acts as an interface for the underlying model's fit method, converting standardized data
        into a format which the MLP can recognize"""
        # Prep the dataset
        self.scaler.fit(x)
        x = self.scaler.transform(x)
        y = (y == 2) # binary data seems to work better
        y = np.array(y, dtype=int)

        mlp_witheld = 0.5

        # Train the AE only on positive samples with some withheld to simulate new data for better generalization
        pos_indices = np.array(np.where(y == 1)[0])
        neg_indices = np.array(np.where(y != 1)[0])
        ae_indices = pos_indices[:int(mlp_witheld * len(pos_indices))]
        mlp_indices = np.concatenate((neg_indices, pos_indices[int(mlp_witheld * len(pos_indices)):]))

        # Fit the AE on its portion of the data
        self._fit_ae(x = x[ae_indices])

        # Get the AE predictions to include with the MLP
        ae_preds = self._predict_ae(x[mlp_indices])

        # Train the MLP on the AE predictions and raw data together
        self._fit_mlp(x = np.hstack((ae_preds, x[mlp_indices])), y=y[mlp_indices])

    def _fit_ae(self, learning_rate=1e-4, loss = 'mse',
            x = None, y = None, batch_size = 128, epochs = 1000, verbose = 0, callbacks = None, validation_split = 0.1,
            validation_data = None, shuffle = True, class_weight = None, sample_weight=None, initial_epoch=0, 
            steps_per_epoch = None, validation_steps = None, validation_batch_size = None, validation_freq = 1, 
            max_queue_size = 10, workers = 1, use_multiprocessing = True):
        """Fit the AE to the data specified"""
        self.ae.compile(optimizer = tf.keras.optimizers.Adam(learning_rate), loss = loss)
        with(tf.device('/CPU:0')):
            self.ae.fit(
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

    def _fit_mlp(self, learning_rate=1e-4, loss = 'binary_crossentropy',
            x = None, y = None, batch_size = 128, epochs = 500, verbose = 0, callbacks = None, validation_split = 0.1,
            validation_data = None, shuffle = True, class_weight = None, sample_weight=None, initial_epoch=0, 
            steps_per_epoch = None, validation_steps = None, validation_batch_size = None, validation_freq = 1, 
            max_queue_size = 10, workers = 1, use_multiprocessing = True):
        """Fit the MLP to the data specified"""
        self.mlp.compile(optimizer = tf.keras.optimizers.Adam(learning_rate), loss = loss)
        self.mlp.fit(
            x,
            y,
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

    def predict(self, data, binary = False):
        """Acts as an interface for the model's prediction method, performs scaling and
        gets data into a format appropriate for testing"""

        # Get the predictions
        preds_ae = self._predict_ae(data)
        preds = self._predict_mlp(data = np.hstack((preds_ae, data)))

        if binary:
            preds = np.squeeze(np.where(preds < 0.5, 1, 0))
        
        return 1 - preds
  
    def _predict_ae(self, data):
        """Get the predictions from AE"""
        return self.ae.predict(data)
    
    def _predict_mlp(self, data):
        """return the predictions from MLP"""
        return self.mlp.predict(data)

    def get_scores(self, data, true):
        """Acts as an interface to get a model's predictions and obtain metrics from them"""
        pred = self.predict(data, binary=True)
        true = np.where(true > 1, 1, 0)

        class_report = classification_report(true, pred, target_names=["neg", "pos"])
        return class_report
