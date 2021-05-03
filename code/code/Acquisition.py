# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 16:27:47 2021

@author: camil
"""

from PIL import ImageGrab
import cv2
import numpy as np
import os
import keyboard
from Variables import *
from Fonctions import *
import time

    
class ImageAcquisition:
    
    """Capture et affiche les images du jeu"""
    
    def __init__(self):
        self.coord_hg = COORD_HG   #bord haut-gauche de la fenetre de jeu
        self.coord_bd = COORD_BD   #bord bas-droit de la fenetre de jeu
        
        #coordonnees du compteur de vitesse
        self.comp_u = COMPTEUR_UNITE
        self.comp_d = COMPTEUR_DIZAINE
        self.comp_c = COMPTEUR_CENTAINE
  
        #Vecteurs utilisés comme inputs lors de l'apprentissage automatique
        self.list_vectors = list(map(
            lambda pixels: get_line(*pixels, 100),
            LIST_VECTORS))
    
        
    def take_entire_screenshot(self):
        """Prend un screenshot de la fenetre de jeu sous forme de np.array"""
    
        return(np.array(ImageGrab.grab(bbox = self.coord_hg + self.coord_bd)))
    
    
    def rogner_screens_compteur(self, full_img):
        """Renvoie un screenshot pour l'unite, la dizaine, la centaine
           du compteur de vitesse a partir du screen de l'ecran complet"""
           
        return [
            full_img[self.comp_c['y1']:self.comp_c['y2'],
                     self.comp_c['x1']:self.comp_c['x2']],
            full_img[self.comp_d['y1']:self.comp_d['y2'],
                     self.comp_d['x1']:self.comp_d['x2']],
            full_img[self.comp_u['y1']:self.comp_u['y2'],
                     self.comp_u['x1']:self.comp_u['x2']]
            ]
                  
      
    def show_live(self):
        """Affiche simplement l'ecran de jeu"""
        
        continuer = True
        while continuer:
            
            #On prend un screen de l'ecran de jeu
            img = self.take_entire_screenshot()
            
            #On recupere sur ce screen le compteur de vitesse
            screens_compteur = self.rogner_screens_compteur(img)
            
            #Delimitation de la route
            img = (is_road(img) | is_finish(img)) *255
            img = np.array(img, dtype = np.uint8)
            
            #Gray -> BGR pour VISUALISER. 
            #Ne pas le faire pour l'apprentissage automatique 
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            #Pour chaque vecteur qui sert de capteur
            for vector in self.list_vectors:
                
                #On va chercher le pixel noir (mur) le plus proche
                vector_bool = list(map(lambda pixel : np.any(img[pixel] ==0), vector))         
                try:
                    index = vector_bool.index(True)
                    pos_mur = vector[index]
                except ValueError:
                    pos_mur = vector[-1]
                    
                #On affiche ce vecteur
                #cv2.line est sous le format de coordonnees (x,y) !
                cv2.line(img,                          #image
                        (vector[0][1], vector[0][0]),  #pixel debut
                        (pos_mur[1], pos_mur[0]),      #pixel fin
                        (0,255,0),                     #couleur
                         3)                            #epaisseur
                
                #On affiche la valeur du vecteur
                cv2.putText(img,                                    #image
                            str(get_distance(vector[0], pos_mur)),  #texte
                            (vector[0][1], vector[0][0]),           #position
                            cv2.FONT_HERSHEY_SIMPLEX,               #font
                            0.5,                                    #taille
                            (0,0,255),                              #couleur
                            2)                                      #epaisseur
            
            #On lit la vitesse pas reconnaissance d'image
            valeur_compteur = reconnaitre_nombre(*screens_compteur)          
            
            #On affiche la vitesse
            cv2.putText(img,                        #image
                        str(valeur_compteur),       #texte
                        (580, 465),                 #position
                        cv2.FONT_HERSHEY_SIMPLEX,   #font
                        1,                        #taille
                        (0,0,255),                  #couleur
                        2)                          #epaisseur
              
            cv2.imshow('live', img)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                continuer = False
                
                
    def show_data(self, file_name):
        """Affiche la video contenue dans un fichier .npy"""
        
        data_file = np.load(file_name, allow_pickle = True)
        
        #Pour chaque frame de la video
        for frame in data_file:
            img = frame[0]           #Image
            inputs = frame[1]        #Inputs
            cv2.imshow('enregistrement', img)
            print(inputs)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        cv2.destroyAllWindows()
    


class InputsAcquisition:
    
    """Capture les inputs claviers"""
    
    def __init__(self):
        #liste des touches utilisees pour jouer
        self.targeted_keys = ['haut', 'bas', 'gauche', 'droite', 'enter', 'backspace', 'q']
    
    
    def get_inputs(self):
        """Renvoie les inputs utilises a l'instant t"""
        
        #On verifie quelles touches de notre liste self.targeted_keys sont utilisees
        used_keys = list()
        for key in self.targeted_keys:
            if keyboard.is_pressed(key):
                used_keys.append(key)
          
        #Si on appuie sur q, on ne prend pas en compte les autres touches
        if 'q' in used_keys:
            return['q']
        
        #Si on appuie sur backspace, on ne prend pas en compte les autres touches
        if 'backspace' in used_keys:
            return ['backspace']
        
        #Si on appuie sur enter, on ne prend pas en compte les autres touches
        if 'enter' in used_keys:
            return ['enter']
    
        return used_keys
            

    
class DataRecorder:
    
    """Enregistre la data : images + inputs"""
    
    def __init__(self):
        
        #dossier d'enregistrement
        self.data_directory = '../data'
        
        #nom du fichier d'enregistrement
        self.record_file = '../data/record'
        
        #Objets d'acquisition
        self.image_acquisition = ImageAcquisition()
        self.inputs_acquisition = InputsAcquisition()
        
        #Cree le dossier s'il n'existe pas.
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)
            
            #initialise l'index du premier fichier d'enregistrement
            self.index = 0
        
        #Si le dossier existe
        else:
            
            #On recherche le fichier de plus grand index
            try:
                
                #On recupere tous les fichier du dossier data_directory
                existing_files = os.listdir(self.data_directory)
                
                #On recupere leurs indexs
                record_indexs = map(lambda file : int(file[6:-4]), existing_files)
                
                #On recupere le max des indexs
                max_index = max(record_indexs)
                
                #On determine l'index du prochain fichier d'enregistrement
                self.index = max_index +1
                
            #S'il n'y a aucun fichier, on initialise l'index du premier.
            except ValueError:
                self.index = 0
            
        
    def record(self):
        """Enregistre la data dans le dossier self.data_directory"""

        data = list()
        
        while True:
            img = self.image_acquisition.take_entire_screenshot()
            inputs = self.inputs_acquisition.get_inputs()
            
            #On ajoute le coupe (image - inputs) a notre data
            data.append((img, inputs))
            
            #'enter' valide la run, 
            #On enregiste la data et incremente l'index pour la suivante
            if inputs == ['enter']:
                np.save(self.record_file + str(self.index), data)
                data = list()
                self.index += 1
            
            #'backspace' signifie qu'on recommence la run,
            #On efface l'enregistrement pour en faire un autre
            elif inputs == ['backspace']:
                data = list()
                
            #'q' sert a couper l'enregistrement
            elif inputs == ['q']:
                break

    
    
####                        ####
####   ESPACE BROUILLON     ####
####                        ####
 
# Gray -> BGR pour VISUALISER. 
# Ne pas le faire pour l'apprentissage automatique
# img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

#   a = list(map(lambda x: np.any(img(x) ==0), get_line((270, 327),(160, 327), 20)))
# try :
#     index = a.index(True)
#     pixel = get_line((270, 327),(160, 327), 20)[index]
# except ValueError:
#     pixel = (160,327)

# for i in get_line((270, 327),(160, 327), 20):
#     pixel_mur = 
#     if np.any(img[i] == 0) :
                    
#         #cv2.line est sous le format (x,y) !
# img = cv2.line(img, (327,270), (pos_mur[1], pos_mur[0]), (0,255,0), 3) 
                    
#         break
# img = cv2.line(img, (50,50), (150,150), (0,255,0), 9)  
# img = ((img[:,:,0] >60)&(img[:,:,1] >60)&(img[:,:,2] >60))*255       
# img3= np.array([[0 for i in range(200)]for j in range(300)], dtype = np.uint8)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.Canny(img, threshold1= 200, threshold2=200)
# lines = cv2.HoughLinesP(img, 1, np.pi/180, 180, 20, 15)
# for line in lines:
#     coords = line[0]
#     cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [255,255,255], 3)
# vertices = np.array([[0,180],[655,180],[655,480],[0,480]])
# mask = np.zeros_like(img)
# cv2.fillPoly(mask, [vertices], 255)
# masked = cv2.bitwise_and(img, mask)