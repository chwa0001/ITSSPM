import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dropout, Dense, Input, LSTM, concatenate, ConvLSTM2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *
from IPython.core.debugger import set_trace
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow_addons as tfa
from tqdm import tqdm
from tensorflow.keras.models import model_from_json


class Sgcn_Lstm():
    def __init__(self, train_x, train_y, AD, AD2, bias_mat_1, bias_mat_2, lr=0.0001, epoach=200, batch_size=10):
        self.train_x = train_x
        self.train_y = train_y
        self.AD = AD
        self.AD2 = AD2
        self.bias_mat_1 = bias_mat_1
        self.bias_mat_2 = bias_mat_2
        self.lr = lr
        self.epoach =epoach
        self.batch_size = batch_size
        self.num_joints = 21 #change to 21 

    def sgcn(self, Input):
        """Temporal convolution"""
        k1 = tf.keras.layers.Conv2D(64, (9,1), padding='same', activation='relu')(Input)
        k =  concatenate([Input, k1], axis=-1)
        """Graph Convolution"""
        
        """first hop localization"""
        x1 = tf.keras.layers.Conv2D(filters=21, kernel_size=(1,1), strides=1, activation='relu')(k)
        logits = x1 
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + self.bias_mat_1)
        gcn_x1 = tf.keras.layers.Lambda(lambda x: tf.einsum('ntvw,ntwc->ntvc', x[0], x[1]))([coefs, x1])
        
        """second hop localization"""
        y1 = tf.keras.layers.Conv2D(filters=21, kernel_size=(1,1), strides=1, activation='relu')(k)
        logits = y1 
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + self.bias_mat_2)
        gcn_y1 = tf.keras.layers.Lambda(lambda x: tf.einsum('ntvw,ntwc->ntvc', x[0], x[1]))([coefs, y1])

        gcn_1 = concatenate([gcn_x1, gcn_y1], axis=-1)
        
        """Temporal convolution"""
        z1 = tf.keras.layers.Conv2D(16, (9,1), padding='same', activation='relu')(gcn_1)
        z1 = Dropout(0.25)(z1)
        z2 = tf.keras.layers.Conv2D(16, (15,1), padding='same', activation='relu')(z1)
        z2 = Dropout(0.25)(z2)
        z3 = tf.keras.layers.Conv2D(16, (20,1), padding='same', activation='relu')(z2)
        z3 = Dropout(0.25)(z3)
        z = concatenate([z1, z2, z3], axis=-1) 
        return z

    def Lstm(self,x):
        x = tf.keras.layers.Reshape(target_shape=(-1,x.shape[2]*x.shape[3]))(x)
        rec = LSTM(80, return_sequences=True)(x)
        rec = Dropout(0.35)(rec)
        rec1 = LSTM(60, return_sequences=True)(rec)
        rec1 = Dropout(0.35)(rec1)
        rec2 = LSTM(40, return_sequences=True)(rec1)
        rec2 = Dropout(0.35)(rec2)
        rec3 = LSTM(40, return_sequences=True)(rec2)
        rec3 = Dropout(0.35)(rec3)
        rec4 = LSTM(40, return_sequences=True)(rec3)
        rec4 = Dropout(0.35)(rec4)
        rec5 = LSTM(80)(rec4)
        rec5 = Dropout(0.35)(rec5)
        out = Dense(1, activation = 'linear')(rec5)
        return out
      
    def train(self):
        seq_input = Input(shape=(None, self.train_x.shape[2], self.train_x.shape[3]), batch_size=None)
        x = self.sgcn(seq_input)
        y = self.sgcn(x)
        y = y + x
        z = self.sgcn(y)
        z = z + y        
        out = self.Lstm(z)
        self.model = Model(seq_input, out)
        self.model.compile(loss=tf.keras.losses.Huber(delta=0.1), experimental_steps_per_execution = 50, optimizer= tf.keras.optimizers.Adam(lr=self.lr))
        best_model_file = "/content/drive/MyDrive/STGCN-rehab-main/best_model_ex2.h5" #change name of weights file here!
        checkpoint = ModelCheckpoint(best_model_file, monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=1)
        tqdm_callback = tfa.callbacks.TQDMProgressBar()
        history = self.model.fit(self.train_x, self.train_y, validation_split=0.2, epochs=self.epoach,batch_size=self.batch_size, callbacks=[checkpoint])
        return history
    
    def model_save(self):
        seq_input = Input(shape=(None, self.train_x.shape[2], self.train_x.shape[3]), batch_size=None)
        x = self.sgcn(seq_input)
        y = self.sgcn(x)
        y = y + x
        z = self.sgcn(y)
        z = z + y        
        out = self.Lstm(z)
        self.model = Model(seq_input, out)
        model_file = self.model.to_json()
        return model_file
      
    def prediction(self, data):
        y_pred = self.model.predict(data)
        return y_pred