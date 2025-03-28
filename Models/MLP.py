import os
import keras
import keras.callbacks
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Input, Dropout
from keras import Model, regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import pickle
import math
from scipy.stats import wilcoxon

class MLP:

    def __init__(self, NDIM, class_weight = None, file_path = './MLP/', model_name = 'mlp'):
        """Initializes a DNN and scaler which can be used in standardized training
        Class weight is unimplemented with pretrained models"""
        self.model_path = file_path + model_name + '.h5'
        self.scaler_path = file_path + model_name + '_scaler.pkl'
        self.pretrained = False
        
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = keras.models.load_model(self.model_path)
            self.scaler = pickle.load(open(self.scaler_path, 'rb'))
            self.class_weight = None
            self.pretrained = True
            
        else: # Train new if can't find necessary components
            input_layer = Input(shape=NDIM)
            
            dense = Dense(NDIM, activation = 'relu', kernel_initializer='he_normal')(input_layer)
            dense = Dropout(0.5)(dense)
            dense = Dense(NDIM, activation = 'relu', kernel_initializer='he_normal')(dense)
            dense = Dropout(0.5)(dense)
            dense = Dense(NDIM, activation = 'relu', kernel_initializer='he_normal')(dense)
            dense = Dropout(0.5)(dense)
            
            output = Dense(3, activation='softmax', kernel_initializer='he_normal')(dense)


            self.model = Model(input_layer, output)
            self.scaler = StandardScaler()
            self.class_weight = class_weight
            
        print('\MLP Model Loaded:')
        print(type(self.model))

    def fit(self, learning_rate=1e-2, loss = 'categorical_crossentropy',
            x = None, y = None, batch_size = 64, epochs = 100, verbose = 0, callbacks = None, validation_split = 0.1,
            validation_data = None, shuffle = True, class_weight = None, sample_weight=None, initial_epoch=0, 
            steps_per_epoch = None, validation_steps = None, validation_batch_size = None, validation_freq = 1, 
            max_queue_size = 10, workers = 1, use_multiprocessing = True):
        """Acts as an interface for the underlying model's fit method, converting standardized data
        into a format which the MLP can recognize"""
        if not self.pretrained:
            self.scaler.fit(x)
            x = self.scaler.transform(x)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)

            # y = np.array(y == 2)
            # print(y)
            y = keras.utils.to_categorical(y)

            self.model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate), loss = loss)
                    

            if callbacks == None:
                callbacks = [
                            #  keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 20, mode='min', restore_best_weights = True)
                            keras.callbacks.ModelCheckpoint(filepath = './chkpnt.h5',
                                                            monitor='val_loss',
                                                            mode = 'min',
                                                            save_best_only=True)
                            ]
            for i in range(5):
                history = self.model.fit(
                    x,
                    y,
                    batch_size,
                    epochs,
                    verbose,
                    callbacks,
                    validation_split,
                    validation_data,
                    shuffle,
                    self.class_weight,
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
            
                import matplotlib.pyplot as plt
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                
                max_val = np.max(np.concatenate([history.history['loss'], history.history['val_loss']]))
                
                plt.title('model loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.ylim([0, max_val])
                # plt.legend(['train', 'val'], loc='upper left')
                plt.savefig('./Loss.png')
                
                self.model.save(self.model_path)

    def predict(self, data, labels = None, binary = False):
        """Acts as an interface for the model's prediction method, performs scaling and
        gets data into a format appropriate for testing - labels are unused"""

        # Get the predictions
        data = StandardScaler().fit_transform(data)
        preds = self.model.predict(data)

        # Return just the class prediction as more of a rating
        pred_1d = []
        for pred in preds:
            pred_1d.append(0 * pred[0] + 1 * pred[1] + 2 * pred[2])
        pred_1d = np.array(pred_1d)

        # Returns binary + or - only instead of raw score
        if binary:
            pred_1d = (pred_1d >= 1.5) # 1.5 is a good cutoff as the highest score is a 3
            
        return pred_1d

    def get_scores(self, data, true):
        """Acts as an interface to get a model's predictions and obtain metrics from them"""
        pred = self.predict(data, binary=True)
        true = np.where(true == 2, 1, 0)

        class_report = classification_report(true, pred, target_names=["neg", "pos"])
        return class_report
