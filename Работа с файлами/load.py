import numpy as np
import cv2 as cv

# параметры цветового фильтра
hsv_min = np.array((59, 119, 17), np.uint8)
hsv_max = np.array((79, 255, 255), np.uint8)

img = cv.imread("image.png")

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV )
# меняем цветовую модель с BGR на HSV
thresh = cv.inRange(hsv, hsv_min, hsv_max )
# применяем цветовой фильтр
# ищем контуры и складируем их в переменную contours
contours, hierarchy = cv.findContours(thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# hierarchy хранит информацию об иерархии
# отображаем контуры поверх изображения
cv.drawContours(img, contours, -1, (255, 0, 0), 2, cv.LINE_AA, hierarchy, 0)
cv.imshow('contours', img)
cv.drawContours(img, contours, -1, (255, 0, 0), 2, cv.LINE_AA, hierarchy, 2)

# выводим итоговое изображение в окно
cv.imshow('All_con', img)
cv.imshow('thresh', thresh)
cv.waitKey()
cv.destroyAllWindows()
