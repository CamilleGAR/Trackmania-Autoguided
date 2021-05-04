# -*- coding: utf-8 -*-
"""
Created on Sat May  1 17:58:27 2021

@author: camil
"""

import numpy as np
import cv2
from Variables import *

def between(array, borne_min, borne_max):
    """Verifie pour chaque valeur d'un array qu'elle soit
    comprise entre les bornes min et max"""
    
    return (array >= borne_min) & (array <= borne_max)

def get_line(pixel_debut, pixel_fin, nb_points):
    """Renvoie une liste de pixels equidistants
       sur la ligne (pixel_debut - pixel_fin)"""
       
    distance_y = pixel_fin[0] - pixel_debut[0]
    distance_x = pixel_fin[1] - pixel_debut[1]
    
    return [(pixel_debut[0] + int(distance_y/nb_points*i),
            pixel_debut[1] + int(distance_x/nb_points*i))
            for i in range(nb_points+1)]

def is_finish(array):
    """Verifie si l'element percu est la ligne d'arrivee.
       On le verifie grace a sa couleur rouge"""
       
    return between(array[:,:,0], 100, 200) \
         & between(array[:,:,1], 30, 70)   \
         & between(array[:,:,2], 30, 70)


def is_road(array):
    """Verifie si l'element percu est la route.
       On le verifie grace a sa couleur grise"""
       
    return (array[:,:,0] >= 60) \
         & (array[:,:,1] >= 60) \
         & (array[:,:,2] >= 60)
         
         
def get_distance(pixel_debut, pixel_fin):
    """Renvoie la distance (en pixels) entre deux pixels"""
    
    x = pixel_fin[1] - pixel_debut[1]
    y = pixel_fin[0] - pixel_debut[0]
    return int(np.sqrt(x**2 + y**2))


def transformation_black_white(image):
    """Transforme une image en noir et blanc
       avec une threshold"""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ignore, threshold_image = cv2.threshold(gray_image, 170, 255, cv2.THRESH_BINARY)
    return np.array(threshold_image, dtype = np.uint8)


def reconnaitre_chiffre(screenshot_chiffre):
    """Reconnaissance d'image sur les chiffres"""
    
    #Transformation du screenshot donne en parametre
    screenshot_processed = transformation_black_white(screenshot_chiffre)
    
    #initialisation
    best_model = {'chiffre' : 0, 'similitude' : 0}
    
    #Comparaison de notre screenshot avec chaque modele
    for chiffre, model in enumerate(MODELES_CHIFFRES):
        
        #Transformation du modele
        model_processed = transformation_black_white(model)
        
        #Calcul des similitudes pour chaque pixel des deux images
        tableau_similitudes = (model_processed == screenshot_processed)
        
        #Total des similitudes
        similitude = np.sum(tableau_similitudes)
        
        #update du model qui correspond le mieux
        if similitude > best_model['similitude'] : 
            best_model['chiffre'] = chiffre
            best_model['similitude'] = similitude
            
    #Si le meilleur model est le 'chiffre inexistant'
    if best_model['chiffre'] == 10: return ''   
    
    #Sinon
    else: return best_model['chiffre']
    
    
def reconnaitre_nombre(*screenshots_chiffres):
    """Concatenation des tous les chiffres a partir des screenshots de chacuns
       Donne finalement un nombre"""
       
    nombre = ''
    for screenshot in screenshots_chiffres:
        nombre += str(reconnaitre_chiffre(screenshot))
        
    #On envite un bug sur un nombre vide ''
    if nombre == '': return 0
    return int(nombre)


def check_diff(nb_1, nb_2, nb_3):
    """Verifie que le nb_2 n'ait pas une difference de +50 par rapport au deux autres
       Sert pour eviter les bugs de reconnaissance de vitesse"""
       
    return (abs(nb_2 - nb_1) > 50) \
         & (abs(nb_2 - nb_3) > 50)