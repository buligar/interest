from PyQt5 import QtWidgets #для создания виджетов
from PyQt5.QtWidgets import QApplication,QMainWindow # для создания приложения и окна

import sys

class Window(QMainWindow):
    def __init__(self): # создание конструктора
        super(Window,self).__init__()
        self.setWindowTitle("Классификация изображений")  # название окна
        self.setGeometry(300, 250, 350, 200)  # размер окна

        self.new_text = QtWidgets.QLabel(self)

        self.main_text = QtWidgets.QLabel(self)
        self.main_text.setText("Это базовая надпись")
        self.main_text.move(100, 100)  # движение текста
        self.main_text.setFixedWidth(200)  # чтобы текст влезал
        self.main_text.adjustSize()  # подстроить ширину объекта(текста)

        self.btn = QtWidgets.QPushButton(self)
        self.btn.move(70, 150)  # движение кнопки
        self.btn.setText("Нажми на меня")
        self.btn.setFixedWidth(200)  # чтобы кнопка влезала
        self.btn.clicked.connect(self.add_label)

    def add_label(self):
        self.new_text.setText("Вторая надпись")
        self.new_text.move(100,50)
        self.new_text.adjustSize()
        print("add")

def application():
    app = QApplication(sys.argv) # набор настроек Создание объекта
    window = Window() # создание окна
    window.show()
    sys.exit(app.exec_()) # для корректного заверщения программы

if __name__=="__main__": # если запускаю файл как основной файл
    application()
