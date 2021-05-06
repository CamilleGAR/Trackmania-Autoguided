# -*- coding: utf-8 -*-
"""
Created on Thu May  6 15:16:13 2021

@author: camil
"""
from DirectKeys import *


class Driver:
    
    """Classe qui sert a gerer les inputs claviers en multithread"""
    
    def __init__(self):
        
        #Dit si la touche doit etre simulee
        self.haut = False
        self.bas = False
        self.droite = False
        self.gauche = False
        
        #Codes correspondant aux touches directionnelles
        self.codes = {'haut': 0xC8,
                      'bas': 0xD0,
                      'gauche': 0xCB,
                      'droite': 0xCD}
        
        def droite(self):
            """Gere la pression de la touche directionnelle droite"""
            
            if self.droite:
                self.PressKey()