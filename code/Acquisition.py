# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 16:27:47 2021

@author: camil
"""

from PIL import ImageGrab
import cv2
import numpy as np


def between(array, borne_min, borne_max):
    """Verifie pour chaque valeur d'un array qu'elle soit
    comprise entre les bornes min et max"""
    
    return (array >= borne_min) & (array <= borne_max)


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
    
    
class ImageAcquisition:
    
    """Capture, affiche, enregistre le jeu"""
    
    def __init__(self):
        self.coord_hg = (0, 30)      #bord haut-gauche de la fenetre de jeu
        self.coord_bd = (655, 510)   #bord bas-droit de la fenetre de jeu
        
        
    def take_screenshot(self):
        """Prend un screenshot de la fenetre de jeu sous forme de np.array"""
    
        return(np.array(ImageGrab.grab(bbox = self.coord_hg + self.coord_bd)))
    
    def show(self):
        """Affiche simplement l'ecran de jeu"""
        
        continuer = True
        while continuer:
            img = self.take_screenshot()
            img = (is_road(img) | is_finish(img)) *255
            img = np.array(img, dtype = np.uint8)
            cv2.imshow('window', img)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                continuer = False
  

              
class InputsAcquisition:
    
    """Capture les inputs claviers"""
    
    def get_inputs(self):
        """"""
        pass
    
    
####                        ####
####   ESPACE BROUILLON     ####
####                        ####
 
#img = ((img[:,:,0] >60)&(img[:,:,1] >60)&(img[:,:,2] >60))*255       
#img3= np.array([[0 for i in range(200)]for j in range(300)], dtype = np.uint8)
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