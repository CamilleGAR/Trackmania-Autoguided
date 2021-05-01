# -*- coding: utf-8 -*-
"""
Created on Sat May  1 17:56:10 2021

@author: camil
"""

from Fonctions import *

COORD_HG = (0, 30)      #bord haut-gauche de la fenetre de jeu
COORD_BD = (655, 510)   #bord bas-droit de la fenetre de jeu
  
#Vecteurs utilisés comme inputs lors de l'apprentissage automatique
LIST_VECTORS = [get_line((270, 327),(160, 327), 60),
                get_line((270, 290),(160, 290), 60),
                get_line((270, 364),(160, 364), 60),
                get_line((290, 260),(130, 190), 60),
                get_line((290, 394),(130, 464), 60),
                get_line((325, 240),(295, 45), 60),
                get_line((325, 414),(295, 609), 60)]
    