import sys
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import np as np
import numpy as np

from skimage.feature import greycomatrix, greycoprops
from skimage import io
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        sc = MplCanvas(self, width=5, height=4, dpi=100)
        image = io.imread('Scratch0.jpg')  # Загрузка изображения
        D = 10  # Расстояние смежности
        Angles = 0
        Distances = np.arange(1, D + 1, 1)
        Angles = [Angles]  # Угол
        glcm = greycomatrix(image, distances=Distances,
                            angles=Angles,  # np.pi/4,np.pi/2,3*np.pi/4,np.pi,5*np.pi/4,3*np.pi/2,7*np.pi/4],
                            levels=256,
                            symmetric=True, normed=True)  # Построение МПС

        Contrast = greycoprops(glcm, 'contrast')  # Текстурный признак Контраст
        Dissimilarity = greycoprops(glcm, 'dissimilarity')  # Текстурный признак несходство
        Homogeneity = greycoprops(glcm, 'homogeneity')  # Текстурный признак Локальная однородность
        Asm = greycoprops(glcm, 'ASM')  # Текстурный признак Угловой второй момент
        Energy = greycoprops(glcm, 'energy')  # Текстурный признак Энергия
        Correlation = greycoprops(glcm, 'correlation')  # Текстурный признак Корреляция

        sc.axes.plot(Distances,Contrast,marker='o',color='Red')
        sc.axes.plot(Distances,Dissimilarity,marker='o',color='Blue')
        sc.axes.plot(Distances,Homogeneity,marker='o',color='Green')
        sc.axes.plot(Distances, Asm, marker='o',color='Yellow')
        sc.axes.plot(Distances, Energy, marker='o',color='Brown')
        sc.axes.plot(Distances, Correlation, marker='o',color='Orange')

        self.setCentralWidget(sc)
        self.show()

app = QtWidgets.QApplication(sys.argv)
w = MainWindow()
app.exec_()
