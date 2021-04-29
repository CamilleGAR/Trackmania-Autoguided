# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 16:27:47 2021

@author: camil
"""

from PIL import ImageGrab
import cv2

class ScreenCapture:
    
    """Capture, affiche, enregistre le jeu"""
    
    def __init__(self):
        self.coord_hg = (0, 30)      #bord haut-gauche de la fenetre de jeu
        self.coord_bd = (655, 510)   #bord bas-droit de la fenetre de jeu
        
        
    def take_screenshot(self):
        """Prend un screenshot de la fenetre de jeu"""
    
        print(ImageGrab.grab(bbox = self.coord_hg + self.coord_bd))
        
    