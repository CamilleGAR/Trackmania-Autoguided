# -*- coding: utf-8 -*-
"""
Created on Tue May  4 11:57:01 2021

@author: camil
"""


import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from Acquisition import *
from Preprocessing import *
from matplotlib import pyplot
import numpy as np
import pandas as pd
import functools
import os
from DirectKeys import PressKey, ReleaseKey

class IaLearner:
    
    """Train un model a partir de nos fichiers .npy"""
    
    def __init__(self, data_directory = '../data_preprocessed_type0'):
        
        file_names = os.listdir(data_directory)
        files_npy = [np.load(f'{data_directory}/{file_name}', allow_pickle = True) 
                     for file_name in file_names]
        self.data = functools.reduce(lambda x,y: np.append(x, y, axis = 0),
                                    files_npy)
        
        #Ici pour prendre connaissance des variables qui sont utiles. Doivent etre setup
        self.keyboard_inputs_str = None  #Doit etre setup
        self.keyboard_inputs_str = None  #Doit etre setup
      
    def setup_inputs_keyboard(self, *inputs):
        """"Setup les inputs claviers (outputs de notre reseaux)"""
        possible_inputs = [['gauche'],['haut'],['droite'],['bas'],
                          ['haut', 'gauche'],['haut', 'droite'],['bas', 'droite'],['bas', 'droite'],
                          []]
        self.keyboard_inputs_str = [possible_inputs[i] for i in inputs]
        self.keyboard_inputs_int = inputs
        
        
    def delete_keyboard_inputs(self, dataframe):
        """Delete les inputs clavier non utilises dans les dataframes"""
        
        possible_inputs = [i for i in range(9)]
        inputs_a_delete = [i for i in possible_inputs if i not in self.keyboard_inputs_int]
        df = dataframe.drop(columns=inputs_a_delete)
        
        return df
        
    def count_true(self, dataframe, column_name):
        """Compte le nombre de True dans une colone d'un dataframe"""
        
        try: 
            count = dataframe[column_name].value_counts()[True]
        except KeyError:
            count = 0
        return count
    
    
    def normalize(self, df, training_data = False):
        """Normalize la data"""
        
        columns_data = [column for column in df 
                        if ('capteur' in column or 'vitesse' in column)]
        columns_boolean = [column for column in df if column not in columns_data]
        
        #Si c'est le training set, on setup pour normalizer le reste
        if training_data:
            self.df_min = df[columns_data].min()
            self.df_max = df[columns_data].max()
            
        #Partie data normalizee
        df_data_norm = df[columns_data]-self.df_min/(self.df_max-self.df_min)
                        
        #Partie boolean inchangee
        df_boolean = df[columns_boolean]
        
        #On concatene la data normalize et les bools
        df_normalized =  df = pd.concat([df_data_norm, df_boolean], axis=1, join="inner")
        
        return df_normalized
            
            
            
        
    
    def setup_data(self, *combinaisons_keyboard):
        """Shuffle, normalize, repartie, ... la data
           Renvoie des dataframes"""
    
        #On choisi les inputs clavier que l'on souhaite utiliser
        self.setup_inputs_keyboard(*combinaisons_keyboard)
        
        data = self.data
        
        #detail data
        capteurs = data[:,0]
        vitesse = data[:,1]
        keyboard_inputs = data[:,2]
        
        #dataframes data
        df_capteurs = pd.DataFrame(pd.DataFrame(data[:,0])[0].to_list(),
                                   columns = [f'capteur{i}' for i in range(len(capteurs[0]))])
        df_vitesse = pd.DataFrame(pd.DataFrame(data[:,1])[0].to_list(), 
                                  columns = ['vitesse'])
        df_keyboard_inputs = pd.DataFrame(pd.DataFrame(data[:,2])[0].to_list(), 
                                          columns = [f'keyboard_inputs{i}' for i in range(len(keyboard_inputs[0]))])
        
        df = pd.concat([df_capteurs, df_vitesse, df_keyboard_inputs], axis=1, join="inner")
        
        #On compte pour chaque inputs clavier
        df_map=map(lambda column: self.count_true(df, column), 
                   [f'keyboard_inputs{i}' for i in combinaisons_keyboard])
        
        #On regarde quel est le minimum d'inputs
        min_inputs = min(df_map)
        
        #On shuffle le dataframe
        df_first_shuffle = df.sample(frac = 1)
        
        #On recupere une fraction des {min_inputs} premiers True de chaque colonne
        df_equilibre = pd.concat(df.loc[df_first_shuffle[f'keyboard_inputs{i}'] == True][:int(min_inputs*0.6)]
                       for i in combinaisons_keyboard)
        
        df_shuffled = df_equilibre.sample(frac = 1)
        
        
        return df_shuffled
        
        
        #On supprime les inputs clavier qu'on utilise pas dans ce training
        df_y_train = self.delete_keyboard_inputs(df_y_train)
        df_y_test = self.delete_keyboard_inputs(df_y_test)
        
        np.take(data,
                np.random.permutation(data.shape[0]),
                axis=0,
                out=data);
        
        capteurs = data[:,0]
        vitesse = data[:,1]
        keyboard_inputs = data[:,2]
        
        #repartition de la data
        X_train, X_test, y_train, y_test = train_test_split(capteurs + vitesse, keyboard_inputs, test_size=0.2)
        
        #Transformation en dataframe
        df_X_train = pd.DataFrame(pd.DataFrame(X_train)[0].to_list())
        df_X_test = pd.DataFrame(pd.DataFrame(X_test)[0].to_list())
        df_y_train = pd.DataFrame(pd.DataFrame(y_train)[0].to_list())
        df_y_test = pd.DataFrame(pd.DataFrame(y_test)[0].to_list())
        

        
        return df_X_train, df_X_test, df_y_train, df_y_test
        
    
    def train(self):
        
        #Recupere les dataframes
        X_train, X_test, y_train, y_test = self.setup_data(0,1,2,4,5) #Combinaisons keyboard qu'on utilise ici
        
        # determine the number of input features
        nb_features = X_train.shape[1]
        nb_outputs = y_train.shape[1]

        model = Sequential()
        
        model.add(Dense(100,
                        activation='relu',
                        kernel_initializer='he_normal',
                        input_shape=(nb_features,)))
        model.add(Dense(25, activation='relu', kernel_initializer='he_normal'))
        model.add(Dropout(0))
        model.add(Dense(25, activation='relu', kernel_initializer='he_normal'))
        model.add(Dropout(0))
        model.add(Dense(25, activation='relu', kernel_initializer='he_normal'))
        model.add(Dropout(0))
        model.add(Dense(25, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(nb_outputs, activation='softmax'))
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        history = model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=1, validation_split=0.3)
        
        pyplot.title('Learning Curves')
        pyplot.xlabel('Epoch')
        pyplot.ylabel('Cross Entropy')
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='val')
        pyplot.legend()
        pyplot.show()
        
        self.model = model
        
        
        
class IaPlayer:
    def __init__(self, model, keyboard_str):
        self.keyboard_str = keyboard_str
        self.model= model
        self.image_acquisition = ImageAcquisition()
        self.preprocessor = Preprocessor()
        
    def play(self):
        
        #Laisse le temps de cliquer sur l'ecran puis force le start
        time.sleep(2)
        PressKey(get_key_code('haut'))
        time.sleep(2.5)
        ReleaseKey(get_key_code('haut'))
        
        #Utilisation du reseau de neurones
        while 1:
            screen = self.image_acquisition.take_entire_screenshot()
            capteurs_et_vitesse = self.preprocessor.preprocessing_frame_type0([screen, None])
            capteurs_et_vitesse = capteurs_et_vitesse[0,0] + capteurs_et_vitesse[0,1]
            keys = self.keyboard_str[np.argmax(self.model.predict([capteurs_et_vitesse]))]
            print(keys)
            for key in ['haut', 'bas', 'gauche', 'droite']:
                if key in keys:
                    PressKey(get_key_code(key))
                else:
                    ReleaseKey(get_key_code(key))

        
        
        
        