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

class MLP:

    def __init__(self, NDIM):
        """Initializes a DNN and scaler which can be used in standardized training"""
        input_layer = Input(shape=NDIM)
        dense = Dense(NDIM, activation = 'relu', kernel_regularizer = regularizers.L1L2(1e-4, 1e-4))(input_layer)
        dense = Dense(NDIM, activation = 'relu', kernel_regularizer = regularizers.L1L2(1e-4, 1e-4))(dense)
        dense = Dense(NDIM, activation = 'relu', kernel_regularizer = regularizers.L1L2(1e-4, 1e-4))(dense)
        output = Dense(1, activation='sigmoid')(dense)

        self.model = Model(input_layer, output)
        self.scaler = StandardScaler()

    def fit(self, learning_rate=1e-4, loss = 'binary_crossentropy',
            x = None, y = None, batch_size = 128, epochs = 500, verbose = 0, callbacks = None, validation_split = 0.1,
            validation_data = None, shuffle = True, class_weight = None, sample_weight=None, initial_epoch=0, 
            steps_per_epoch = None, validation_steps = None, validation_batch_size = None, validation_freq = 1, 
            max_queue_size = 10, workers = 1, use_multiprocessing = False):
        """Acts as an interface for the underlying model's fit method, converting standardized data
        into a format which the MLP can recognize"""
        self.scaler.fit(x)
        x = self.scaler.transform(x)

        y = np.array(y == 2)
        # print(y)

        self.model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate), loss = loss)

        self.model.fit(
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
        data = self.scaler.transform(data)
        preds = self.model.predict(data)

        # Returns binary + or - only instead of raw score
        if binary:
            preds = np.where(preds >= 0.5, 1, 0)

        # print(preds)
        return preds

    def get_scores(self, data, true):
        """Acts as an interface to get a model's predictions and obtain metrics from them"""
        pred = self.predict(data, binary=True)
        true = np.where(true > 1, 1, 0)

        class_report = classification_report(true, pred, target_names=["neg", "pos"])
        return class_report
