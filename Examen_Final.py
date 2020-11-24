# Examen Final -- Examen_Final.py
# Leidy Carolina Pulido Feo
# Procesamiento de Imágenes y visión

import numpy as np
from Bandera import * # Importa la clase "Bandera"
from hough import hough
import cv2
import os

if __name__ == '__main__':

    path = 'J:/Proc.Imagenes/Imagenes'
    image_name = input(' flag1.png \n flag2.png \n flag3.png \n flag4.png \n flag5.png \n '
                       'Basado en la lista anterior, ingrese el nombre de la imagen de la bandera que desea procesar:  ')     # Variable que almacena la ruta de la imagen a trabajar
    New_Image = Bandera(path, image_name)    # New_Image es la instancia de la clase colorImage
    #J:/Proc.Imagenes/Imagenes/flag1.png
    Num_colors  = New_Image.Colores()
    print("El número de colores de la bandera", image_name, "es:", Num_colors)
    New_Image.Porcentaje()
    New_Image.Orientacion()
