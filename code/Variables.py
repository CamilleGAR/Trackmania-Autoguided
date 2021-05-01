# -*- coding: utf-8 -*-
"""
Created on Sat May  1 17:56:10 2021

@author: camil
"""

import cv2

COORD_HG = (0, 30)      #bord haut-gauche de la fenetre de jeu
COORD_BD = (655, 510)   #bord bas-droit de la fenetre de jeu


#zone du compteur de vitesse
COMPTEUR_UNITE =    {'y1': 485-COORD_HG[1], 'y2': 510-COORD_HG[1],
                     'x1': 635, 'x2': 650}
COMPTEUR_DIZAINE =  {'y1': 485-COORD_HG[1], 'y2': 510-COORD_HG[1],
                     'x1': 620, 'x2': 635}
COMPTEUR_CENTAINE = {'y1': 485-COORD_HG[1], 'y2': 510-COORD_HG[1],
                     'x1': 605, 'x2': 620}

#Vecteurs utilisés comme inputs lors de l'apprentissage automatique
LIST_VECTORS = [((270, 327),(160, 327)),
                ((270, 290),(160, 290)),
                ((270, 364),(160, 364)),
                ((290, 260),(130, 190)),
                ((290, 394),(130, 464)),
                ((325, 240),(295, 45)),
                ((325, 414),(295, 609))]
 

#Image modeles des chiffres 
MODELES_CHIFFRES = [cv2.imread(f'chiffre{x}.jpg') for x in range(10)] \
                  +[cv2.imread('chiffreInexistant.jpg')]
