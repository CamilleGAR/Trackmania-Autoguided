# -*- coding: utf-8 -*-
"""
Created on Sat May  1 17:58:27 2021

@author: camil
"""


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
         