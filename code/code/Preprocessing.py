# -*- coding: utf-8 -*-
"""
Created on Sat May  1 17:53:12 2021

@author: camil
"""

import os
from Variables import *
from Fonctions import *

class Preprocessor:
    
    """Sert a epurer la donnee"""
    
    def __init__(self, data_directory = '../data'):
        
        #Fichier data d'originie
        self.data_directory = data_directory
        
        #Vecteurs utilisés comme inputs lors de l'apprentissage automatique
        self.list_vectors = list(map(
            lambda pixels: get_line(*pixels, 100),
            LIST_VECTORS))
    
    def preprocessing_0(self):
        """Methode de preprocessing numero 0"""
        
        data_files = os.listdir(self.data_directory)
        for file in data_files :
            print(file)