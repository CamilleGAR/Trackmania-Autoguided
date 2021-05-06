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
import threading
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
        
        #Ici pour prendre connaissance des variables qui sont utiles. 
        #Doivent etre setup au cours du programme
        self.keyboard_inputs_str = None
        self.keyboard_inputs_int = None  
        self.keyboard_columns_df = None
        self.df_min = None
        self.df_max = None
      
        
    def setup_inputs_keyboard(self, *inputs):
        """"Setup les inputs claviers (outputs de notre reseaux)"""
        possible_inputs = [['gauche'],['haut'],['droite'],['bas'],
                          ['haut', 'gauche'],['haut', 'droite'],['bas', 'droite'],['bas', 'droite'],
                          []]
        self.keyboard_inputs_str = [possible_inputs[i] for i in inputs]
        self.keyboard_inputs_int = inputs
        self.keyboard_columns_df = [f'keyboard_inputs{i}' for i in inputs]
        
        
    def delete_keyboard_inputs(self, dataframe):
        """Delete les inputs clavier non utilises dans les dataframes"""
        
        actual_inputs = [column for column in dataframe if 'keyboard_inputs' in column]
        columns_a_delete = [column for column in actual_inputs if column not in self.keyboard_columns_df]
        df = dataframe.drop(columns=columns_a_delete)
        
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
        
        #Si c'est le training set, on setup pour normalizer le reste
        if training_data:
            self.df_min = df.min()
            self.df_max = df.max()
            
        #dataframe normalized
        df_normalized = (df-self.df_min)/(self.df_max-self.df_min)
        
        return df_normalized
    
    
    def get_attributs_predict(self):
        """Donne les elements importants pour les predictions"""
    
        return {'model': self.model,
                'keyboard_inputs_str': self.keyboard_inputs_str, 
                'df_min': self.df_min,
                'df_max': self.df_max}
    
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
        
        #On shuffle pour avoir un ensemble homogene
        df_shuffled = df_equilibre.sample(frac = 1)

        #On supprime les inputs clavier qu'on utilise pas dans ce training
        df_deleted = self.delete_keyboard_inputs(df_shuffled)

        #On recupere le nom des colonnes
        columns_data = [column for column in df_deleted
                        if ('capteur' in column or 'vitesse' in column)]
        columns_boolean = [column for column in df_deleted if column not in columns_data]
        
        #On repartie en dataset de training et de test
        X_train, X_test, y_train, y_test = train_test_split(df_deleted[columns_data],    #Inputs
                                                            df_deleted[columns_boolean], #Outputs
                                                            test_size=0.2)               #Repartition
        
        #On normalize les valeurs.
        X_train = self.normalize(X_train, training_data = True)
        X_test = self.normalize(X_test)
        
        return X_train, X_test, y_train, y_test
        
    
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
        model.add(Dropout(0.2))
        model.add(Dense(100, activation='relu', kernel_initializer='he_normal'))
        model.add(Dropout(0.2))
        model.add(Dense(100, activation='relu', kernel_initializer='he_normal'))
        model.add(Dropout(0.2))
        model.add(Dense(100, activation='relu', kernel_initializer='he_normal'))
        model.add(Dropout(0.2))
        model.add(Dense(100, activation='relu', kernel_initializer='he_normal'))
        model.add(Dropout(0.2))
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
    def __init__(self, model, keyboard_inputs_str, df_min, df_max):
        self.keyboard_inputs_str = keyboard_inputs_str
        self.model= model
        self.df_min = df_min
        self.df_max = df_max
        self.image_acquisition = ImageAcquisition()
        self.preprocessor = Preprocessor()
        
    def play(self, show = False):
        
        #Thread pour visualiser les capteurs
        if show:
            thread = threading.Thread(target=self.image_acquisition.show_live86)
            thread.start()
        
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
            capteurs_et_vitesse_normalized = (capteurs_et_vitesse-self.df_min)/(self.df_max-self.df_min)
            num_input = np.argmax(self.model.predict([capteurs_et_vitesse_normalized.to_list()]))
            keys = self.keyboard_inputs_str[num_input]
            print(keys)
            for key in ['haut', 'bas', 'gauche', 'droite']:
                if key in keys:
                    PressKey(get_key_code(key))
                else:
                    ReleaseKey(get_key_code(key))

        
        
        
        