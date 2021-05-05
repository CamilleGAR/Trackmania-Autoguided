# -*- coding: utf-8 -*-
"""
Created on Tue May  4 11:57:01 2021

@author: camil
"""


import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
import functools
import os

class IaLearner:
    
    """Train un model a partir de nos fichiers .npy"""
    
    def __init__(self, data_directory = '../data_preprocessed_type0'):
        
        file_names = os.listdir(data_directory)
        files_npy = [np.load(f'{data_directory}/{file_name}', allow_pickle = True) 
                     for file_name in file_names]
        self.data = functools.reduce(lambda x,y: np.append(x, y, axis = 0),
                                    files_npy)
      
    def delete_keyboard_inputs0(self, dataframe):
        """Methode 0 qui delete certains inputs claviers
           D'autres methodes pourront etre creees si on veut travailler avec d'autres compositions d'inputs"""
           
        possible_inputs = [['gauche'],['haut'],['droite'],['bas'],
                                ['haut', 'gauche'],['haut', 'droite'],['bas', 'droite'],['bas', 'droite'],
                                []]
        df = dataframe.drop(columns=[3, 6, 7])
        self.keyboard_inputs = [possible_inputs[i] for i in [0,1,2,4,5,8]]
           
    def setup_data(self):
        """Shuffle, normalize, repartie, ... la data
           Renvoie des dataframes"""
        
        data = self.data
        np.take(data,
                np.random.permutation(data.shape[0]),
                axis=0,
                out=data);
        
        capteurs = data[:,0]
        vitesse = data[:,1]
        keyboard_inputs = data[:,2]
        
        #repartition de la data
        X_train, X_validation, y_train, y_validation = train_test_split(capteurs + vitesse, keyboard_inputs, test_size=0.33)
        
        #Transformation en dataframe
        df_X_train = pd.DataFrame(pd.DataFrame(X_train)[0].to_list())
        df_X_validation = pd.DataFrame(pd.DataFrame(X_validation)[0].to_list())
        df_y_train = pd.DataFrame(pd.DataFrame(y_train)[0].to_list())
        df_y_validation = pd.DataFrame(pd.DataFrame(y_validation)[0].to_list())
        
        #On supprime les inputs clavier qu'on utilise pas dans ce training
        df_y_train = self.delete_keyboard_inputs0(df_y_train)
        df_y_validation = self.delete_keyboard_inputs0(df_y_validation)
        
        return df_X_train, df_X_validation, df_y_train, df_y_validation
        
    def train(self):
        
        #Recupere les dataframes
        X_train, X_validation, y_train, y_validation = self.setup_data()
        
        print(X_train.shape)
        print(y_train.shape)
        # determine the number of input features
        nb_features = X_train.shape[1]
        nb_outputs = y_train.shape[1]

        model = Sequential()
        
        model.add(Dense(100,
                        activation='relu',
                        kernel_initializer='he_normal',
                        input_shape=(nb_features,)))
        model.add(Dense(100, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(100, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(nb_outputs, activation='softmax'))
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=1)
        
        loss, acc = model.evaluate(X_validation, y_validation, verbose=0)
        print('Test Accuracy: %.3f' % acc)
        row = [150,23,53,43,15,65,150,65]
        yhat = model.predict([row])
        print('Predicted: %s (class=%d)' % (yhat, np.argmax(yhat)))
        
        self.model = model
        
        