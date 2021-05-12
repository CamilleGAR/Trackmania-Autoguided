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
    
    
    def get_binary_inputs(self, str_inputs):
        """Transformation des inputs en liste binaire"""
        
        return [str_inputs == ['gauche'],
                str_inputs == ['haut', 'gauche'],
                str_inputs == ['haut'],
                str_inputs == ['haut', 'droite'],
                str_inputs == ['droite']]
    

    
    def preprocessing_frame_type0(self, frame):
        """Methode 0 du preprocessing sur une unique image"""
        
        img = frame[0]
        str_inputs = frame[1]
        
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
        vitesse = [reconnaitre_nombre(*screens_compteur)]       
        
        #Si on a les inputs (dataset de training)
        if str_inputs != None:
            #Transformation des inputs en liste binaire
            binary_inputs = self.get_binary_inputs(str_inputs)
            
            #les inputs dont on va avoir besoin pour cette methode.
            #Sert juste a verifier la validite des inputs, mais on renvoie la liste complete.
            current_inputs = [binary_inputs[i] for i in [0,1,2,3,4]]
            
            #Si aucun input n'est valide, on ne renvoie pas la frame
            if not any(current_inputs):
                return None
    
            return np.array([[vector_outputs, vitesse, binary_inputs]])
        
        #Si on ne les a pas (prediction)
        else:
            return np.array([[vector_outputs, vitesse]], dtype = object)
    
        
    def delete_outliers(self, processed_data_array):
        """Delete les frames avec des valeurs de capteurs incohérentes"""
        
        del_outliers = processed_data_array
        
        #Quand on delete une ligne, il y en une de moins a compter
        malus_index = 0
        
        #On compare chaque frame avec celle d'avant et celle d'apres
        for index, frames in enumerate(zip(processed_data_array[:,1][:-2],processed_data_array[:,1][1:-1],processed_data_array[:,1][2:])):
            
            #On verifie que la difference de vitesse n'est pas absurde
            if (abs(frames[1][0] - frames[0][0]) > 50) \
             & (abs(frames[1][0] - frames[2][0]) > 50):
                del_outliers = np.delete(del_outliers, index+1 -malus_index, 0)
                malus_index += 1
                
        return del_outliers
            
    
    def delete_speed0(self, processed_data_array):
        """Delete les frame ou la voiture n'avance pas"""
        
        del_speed0 = processed_data_array
    
        #Quand on delete une ligne, il y en une de moins a compter
        malus_index = 0
        
        for index, frame in enumerate(processed_data_array[:,1]):
            if frame == [0]:
                del_speed0 = np.delete(del_speed0, index -malus_index, 0)
                malus_index += 1
                
        return del_speed0
    
    def delete_first_last_frames(self, processed_data_array, nb_frames):
        """Deletes de nb_frames premieres et dernieres frames"""
        
        return processed_data_array[50:-50]
    
        
    def preprocessing(self, preprocessing_directory = '../data_preprocessed_type0'):
        """Methode de preprocessing numero 0"""
        
        #Cree le dossier s'il n'existe pas.
        if not os.path.exists(preprocessing_directory):
            os.makedirs(preprocessing_directory)
            
        #Pour chaque enregistrement
        data_files = os.listdir(self.data_directory)
        for file_name in data_files :
            data = np.load(f'{self.data_directory}/{file_name}', allow_pickle = True)
            
            #
            processed_data_array = np.array([[[0,0,0,0,0,0,0],           #capteurs
                                              [0],                       #vitesse
                                              [0,0,0,0,0,0,0,0,0]]])     #inputs
            
            #Preprocessing de chaque frame
            for frame in data:
                
                #On recupere la data sous une forme utilisable
                frame_processed_data = self.preprocessing_frame_type0(frame)
                
                #On l'ajoute a notre array
                if type(frame_processed_data) != type(None):
                    processed_data_array = np.append(processed_data_array, 
                                                     frame_processed_data,
                                                     axis = 0)
            #Delete les frames ininteressantes
            processed_data_array = self.delete_outliers(processed_data_array)
            processed_data_array = self.delete_speed0(processed_data_array)
            processed_data_array = self.delete_first_last_frames(processed_data_array, 50)

            #On enregistre la data exploitable
            np.save(f'{preprocessing_directory}/{file_name[:-4]}_preprocessed', processed_data_array)