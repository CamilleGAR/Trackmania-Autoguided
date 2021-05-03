# -*- coding: utf-8 -*-
"""
Created on Sat May  1 17:53:12 2021

@author: camil
"""

import os
import numpy as np
from Variables import *
from Fonctions import *
from Acquisition import *

class Preprocessor:
    
    """Sert a epurer la donnee"""
    
    def __init__(self, data_directory = '../data'):
        
        #Fichier data d'originie
        self.data_directory = data_directory
        
        #Vecteurs utilisés comme inputs lors de l'apprentissage automatique
        self.list_vectors = list(map(
            lambda pixels: get_line(*pixels, 100),
            LIST_VECTORS))
    
        self.image_acquisition = ImageAcquisition()
    
    
    def preprocessing_frame_type0(self, frame):
        """
    def preprocessing(self, preprocessing_directory = '../data_preprocessed_type0'):
        """Methode de preprocessing numero 0"""
        
        #Cree le dossier s'il n'existe pas.
        if not os.path.exists(preprocessing_directory):
            os.makedirs(preprocessing_directory)
            
        #Pour chaque enregistrement
        data_files = os.listdir(self.data_directory)
        for file_name in data_files :
            data = np.load(f'{self.data_directory}/{file_name}', allow_pickle = True)
            
            for frame in data :
                        
                img = frame[0]
                #On recupere sur ce screen le compteur de vitesse
                screens_compteur = self.image_acquisition.rogner_screens_compteur(img)
                
                #Delimitation de la route
                img = (is_road(img) | is_finish(img)) *255
                img = np.array(img, dtype = np.uint8)
                
                #Pour chaque vecteur qui sert de capteur
                vector_outputs = list()
                for vector in self.list_vectors:
                    
                    #On va chercher le pixel noir (mur) le plus proche
                    vector_bool = list(map(lambda pixel : np.any(img[pixel] ==0), vector))         
                    try:
                        index = vector_bool.index(True)
                        pos_mur = vector[index]
                    except ValueError:
                        pos_mur = vector[-1]
                        
                    #On ajoute la valeur du vecteur a notre liste
                    vector_outputs.append(get_distance(vector[0], pos_mur))
                
                #On lit la vitesse par reconnaissance d'image
                vitesse = reconnaitre_nombre(*screens_compteur)          
                
                print(vector_outputs, vitesse)

            
            # np.save(f'{preprocessing_directory}/{file_name[:-4]}_preprocessed', 
            #         data)