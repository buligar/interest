import np as np
import numpy as np
import sys
import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
from skimage.feature import greycomatrix, greycoprops
from PyQt5 import QtCore, QtWidgets, uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenuBar, QMenu, QFileDialog, QPushButton, QLabel, QHBoxLayout,QListWidgetItem
from PyQt5.QtGui import QPixmap

matplotlib.use('QT5Agg')


class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.ui=uic.loadUi('GLCM.ui', self)
        self.fig = plt.figure()  # Для отображения графиков
        self.fig.subplots_adjust(hspace=0.4, wspace=0.4)  # Суб-график
        self.plotWidget = FigureCanvas(self.fig)
        lay = QtWidgets.QVBoxLayout(self.GraphWidget)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.plotWidget)
        self.addFunctions() #вызов функций кнопок


    def listitemclicked(self, item):
        self.file_path =  self.listWidget.selectedItems()[0].text() # Путь к выбранному файлу
        # Отображение изображения на label
        self.ui.label.setScaledContents(True)
        pixmap=QPixmap(self.file_path)
        self.ui.label.setPixmap(pixmap)
        self.ui.label.repaint()

        print(self.file_path)

    def addFunctions(self):
        # Menu
        self.menuBar = QMenuBar(self)
        self.setMenuBar(self.menuBar)
        fileMenu = QMenu("&Файл", self)
        self.menuBar.addMenu(fileMenu)
        fileMenu.addAction('Открыть',self.action_clicked)
        fileMenu.addAction('Сохранить', self.action_clicked)
        # load image
        self.listWidget.itemClicked.connect(self.listitemclicked)
        # GLCM
        self.pushButton.clicked.connect(lambda: self.glcm())
        # Angles
        # self.angles()



    @QtCore.pyqtSlot()
    def action_clicked(self):
        action = self.sender()
        if action.text() == "Открыть":
            try:
                self.files_path = QFileDialog.getOpenFileNames(self)[0] # извлечение изображений из Dialog окна
                self.listWidget.addItems(self.files_path) # добавление изображений в список

            except FileNotFoundError:
                print("No such file")
        elif action.text() == "Сохранить":
            fname = QFileDialog.getSaveFileName(self)[0]
            try:
                f = open(fname,'w')
                text = self.text_edit.toPlainText()
                f.write(text)
                f.close()
            except FileNotFoundError:
                print("No such file")

    # def angles(self):
    #     self.radioButton0=self.radioButton0.changeEvent(self.onClicked)
    #     print(self.radioButton0)

    # def angles(self):
    #     self.angle_1 = self.angle_1.isChecked()
    #     self.angle_2 = self.angle_2.isChecked()
    #     self.angle_3 = self.angle_3.isChecked()
    #     self.angle_4 = self.angle_4.isChecked()
    #     self.angle_5 = self.angle_5.isChecked()
    #     self.angle_6 = self.angle_6.isChecked()
    #     self.angle_7 = self.angle_7.isChecked()
    #     self.angle_8 = self.angle_8.isChecked()
    #     if self.angle_1 == 1:
    #         Angles_0=[0]
    #         print("angle1", Angles_0)
    #     elif self.angle_1 == 0:
    #         Angles_0=[]
    #     if self.angle_2 == 1:
    #         Angles_45=[np.pi/4]
    #         print("angle2",Angles_45)
    #     elif self.angle_2 == 0:
    #         Angles_45=[]
    #     if self.angle_3 == 1:
    #         Angles_90 = [np.pi/2]
    #         print("angle3", Angles_90)
    #     elif self.angle_3 == 0:
    #         Angles_90=[]
    #     if self.angle_4 == 1:
    #         Angles_135 = [3*np.pi/4]
    #         print("angle4", Angles_135)
    #     elif self.angle_4 == 0:
    #         Angles_135=[]
    #     if self.angle_5 == 1:
    #         Angles_180 = [np.pi]
    #         print("angle5", Angles_180)
    #     elif self.angle_5 == 0:
    #         Angles_180=[]
    #     if self.angle_6 == 1:
    #         Angles_225 = [5*np.pi/4]
    #         print("angle6", Angles_225)
    #     elif self.angle_6 == 0:
    #         Angles_225=[]
    #     if self.angle_7 == 1:
    #         Angles_270 = [3*np.pi/2]
    #         print("angle7", Angles_270)
    #     elif self.angle_7 == 0:
    #         Angles_270=[]
    #     if self.angle_8 == 1:
    #         Angles_315 = [7*np.pi/4]
    #         print("angle8", Angles_315)
    #     elif self.angle_8 == 0:
    #         Angles_315=[]
    #
    #     self.Angles=Angles_0+Angles_45+Angles_90+Angles_135+Angles_180+Angles_225+Angles_270+Angles_315
    #     # self.Ang=self.Ang_0+self.Ang_45+self.Ang_90+self.Ang_135+self.Ang_180+self.Ang_225+self.Ang_270+self.Ang_315
    #     print(self.Angles)

    def glcm(self):
        np.set_printoptions(edgeitems=1000) # для отображения полной матрицы изображения в окне вывода
        grayscale=cv2.imread(self.file_path,0) # Преобразование в оттенки серого
        self.value = self.spinBox.value() # Значение Расстояние смежности в spinbox
        Distances = np.arange(1, self.value + 1, 1) # Расстояние смежности на графике
        glcm = greycomatrix(grayscale, distances=Distances,
                            angles=[0,np.pi/4,np.pi/2,3*np.pi/4],
                            levels=256,
                            symmetric=True, normed=True)  # Построение МПС
        # print(glcm)

        Contrast = greycoprops(glcm, 'contrast')  # Текстурный признак Контраст
        Dissimilarity = greycoprops(glcm, 'dissimilarity')  # Текстурный признак несходство
        Homogeneity = greycoprops(glcm, 'homogeneity')  # Текстурный признак Локальная однородность
        Asm = greycoprops(glcm, 'ASM')  # Текстурный признак Угловой второй момент
        Energy = greycoprops(glcm, 'energy')  # Текстурный признак Энергия
        Correlation = greycoprops(glcm, 'correlation')  # Текстурный признак Корреляция

        plt.subplot(2, 3, 1)
        plt.grid(axis='both')
        plt.title("Контраст")
        plt.xticks([i for i in range(0, max(Distances + 1))])
        plt.plot(Distances, Contrast, marker='o')

        plt.subplot(2, 3, 2)
        plt.grid(axis='both')
        plt.title("Несходство")
        plt.xticks([i for i in range(0, max(Distances + 1))])
        plt.plot(Distances, Dissimilarity, marker='o')

        plt.subplot(2, 3, 3)
        plt.grid(axis='both')
        plt.title("Локальная однородность")
        plt.xticks([i for i in range(0, max(Distances + 1))])
        plt.plot(Distances, Homogeneity, marker='o')

        plt.subplot(2, 3, 4)
        plt.grid(axis='both')
        plt.title("Угловой второй момент")
        plt.xticks([i for i in range(0, max(Distances + 1))])
        plt.plot(Distances, Asm, marker='o')

        plt.subplot(2, 3, 5)
        plt.grid()
        plt.title("Энергия")
        plt.xticks([i for i in range(0, max(Distances + 1))])
        plt.plot(Distances, Energy, marker='o')

        plt.subplot(2, 3, 6)
        plt.grid(axis='both')
        plt.title("Корреляция")
        plt.xticks([i for i in range(0, max(Distances + 1))])
        plt.plot(Distances, Correlation, marker='o')

        self.Anglelegend = [0,45,90,135]

        plt.figlegend(self.Anglelegend)
        plt.show()


if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())

