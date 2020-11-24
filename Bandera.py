#Examen final -- Bandera.py
# Leidy Carolina Pulido Feo
# Procesamiento de Imágenes y visión

import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from hough import hough
import cv2
import os


class Bandera:   # Creación de la clase Bandera

    def __init__(self, path, image_name):    # Constructor -- Recibe, carga y almacena la imagen
        self.path = path
        self.image_name = image_name
        self.path_file = os.path.join(self.path, self.image_name)
        self.image = cv2.imread(self.path_file)  # Carga la imagen de la dirección ruta y la almacena en la valiable image

    def Colores(self):   # Método Colores
        self.n_colors = 4

        # Convert to floats instead of the default 8 bits integer coding. Dividing by
        # 255 is important so that plt.imshow behaves works well on float data (need to
        # be in the range [0-1])
        self.image = np.array(self.image, dtype=np.float64) / 255

        rows, cols, ch = self.image.shape
        assert ch == 3
        self.image_array = np.reshape(self.image, (rows * cols, ch))

        self.image_array_sample = shuffle(self.image_array, random_state=0)[:10000]

        self.model = KMeans(n_clusters=self.n_colors, random_state=0).fit(self.image_array_sample)

        self.labels = self.model.predict(self.image_array)
        self.colores = set(self.labels) #Elimina componentes repetidoas
        self.cc = len(self.colores) #Tamañi - cantidad de colores
        return self.cc

    def Porcentaje(self):  # Método Porcentaje
        self.contar = [0,0,0,0]
        for i in self.labels:
            if i == 0:
                self.contar[0] = self.contar[0] + 1
            elif i == 1:
                self.contar[1] = self.contar[1] + 1
            elif i == 2:
                self.contar[2] = self.contar[2] + 1
            else:
                self.contar[3] = self.contar[3] + 1

        self.porc_col_1 = (self.contar[0] / len(self.labels) * 100)
        if (self.porc_col_1 != 0):
            print("El porcentaje del primer color es : ", self.porc_col_1)
        self.porc_col_2 = (self.contar[1] / len(self.labels) * 100)
        if (self.porc_col_2 != 0):
            print("El porcentaje del segundo color es : ", self.porc_col_2)
        self.por_col_3 = (self.contar[2] / len(self.labels) * 100)
        if (self.por_col_3 != 0):
            print("El porcentaje del tercer color es : ", self.por_col_3)
        self.por_col_4 = (self.contar[3] / len(self.labels) * 100)
        if (self.por_col_4 != 0):
            print("El porcentaje del cuarto color es : ", self.por_col_4)

    def Orientacion(self):  #Método Orientación
        self.high_thresh = 300
        self.bw_edges = cv2.Canny(self.image, self.high_thresh * 0.3, self.high_thresh, L2gradient=True)

        self.hough = hough(self.bw_edges)  # Se le pasa la imagen con bordes
        accumulator = self.hough.standard_HT()  # Se crea el acumulador

        acc_thresh = 50  # Mínimo 50 pixeles que pasen por allí
        N_peaks = 11
        nhood = [25, 9]  # Tamaño de la ventana
        peaks = hough.find_peaks(accumulator, nhood, acc_thresh,N_peaks)  # Encuentra los picos, con valor de ro y el ángulo

        [_, cols] = self.image.shape[:2]
        image_draw = np.copy(self.image)
        for i in range(len(peaks)):
            rho = peaks[i][0]
            theta_ = hough.theta[peaks[i][1]]
            print(theta_)
            theta_pi = np.pi * theta_ / 180
            theta_ = theta_ - 180
            a = np.cos(theta_pi)
            b = np.sin(theta_pi)
            x0 = a * rho + hough.center_x
            y0 = b * rho + hough.center_y
            c = -rho
            # Encontrar dos puntos de la recta
            x1 = int(round(x0 + cols * (-b)))
            y1 = int(round(y0 + cols * a))
            x2 = int(round(x0 - cols * (-b)))
            y2 = int(round(y0 - cols * a))


            #Con base en los ángulos (THETA) se sabe la orientación de la línea y se pueden determinar si son verticales horizontales o miztas
            if np.abs(theta_) < 80:
                image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 255, 255], thickness=2)
            elif np.abs(theta_) > 100:
                image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [255, 0, 255], thickness=2)
            else:
                if theta_ > 0:
                    image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 255, 0], thickness=2)
                    print("Línea Horizontal")
                else:
                    image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 0, 255], thickness=2)
                    print("Línea Horizontal")

        cv2.imshow("frame", self.bw_edges)
        cv2.imshow("lines", image_draw)
        cv2.waitKey(0)
