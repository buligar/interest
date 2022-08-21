import math

import pyautogui

screenWidth, screenHeight = pyautogui.size()
print(screenWidth,screenHeight)

currentMouseX, currentMouseY = pyautogui.position()
print(currentMouseX,currentMouseY)

pyautogui.moveTo(100,150)

def draw_circle_by_mouse(x,y,radius):
    pyautogui.moveTo(x,y)
    i=0
    sectors = 50
    for i in range(0,999999999999999999999):
        i = i + math.pi / sectors
        xm = x + math.sin(i) * radius
        ym  = y + math.cos(i) * radius
        pyautogui.moveTo(xm,ym)

draw_circle_by_mouse(400,400,200)