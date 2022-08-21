import numpy
import numpy as np
import matplotlib.pyplot as plt


N = 100 # кол-во шагов
STEP=1/(N-1) # размер шага
x = np.linspace(0, 1, N) # определение области значения от 0 до 1 с шагом N
F = lambda x: (x**2)*numpy.exp(x) # Функция e^2*exp(x)
F1 = lambda x: (x*(x+2)*numpy.exp(x)) # 1 производная функция, вычисленная аналитически
F2 = lambda x: (x**2+4*x+2)*numpy.exp(x) # 2 производная функции, вычисленная аналитически
for i in range(1,N+1):
    XRAY= lambda i: (i-1)*STEP
    x=XRAY(i)
    Y1RAY = lambda i:F(x)
    Y2RAY = lambda i:F1(x)
    Y4RAY = lambda i:F2(x)

#1 производная функция, вычисленная численно
for i in range(1,N+1):
    if i == 1:
        Y3RAY =lambda i:(-3*F(XRAY(i))+4*F(XRAY(i+1))-F(XRAY(i+2)))/(2*STEP)
    else:
        Y3RAY =lambda i:(F(XRAY(N-2))-4*F(XRAY(N-1))+3*F(XRAY(N)))/(2*STEP)
for i in range(2,N):
    Y3RAY =lambda i:(F(XRAY(i+1))-F(XRAY(i-1)))/(2*STEP)
    razn = Y2RAY(i)-Y3RAY(i) #разность между функциями вычисленными аналитически и численно
    Ekm = numpy.abs(razn) # максимальная погрешность численного дифференцирования
    ek = numpy.sqrt((1 / (N + 1)) * ((razn)**2)) # среднеквадратичная погрешность численного дифференцирования
print('Ekm=',Ekm,'Ek',ek)

#2 производная функция, вычисленная численно
for i in range(1,N+1):
    if i == 1:
        Y5RAY =lambda i:(12 * F(XRAY(i)) - 30 * F(XRAY(i+1)) + 24 * F(XRAY(i+2))-6 * F(XRAY(i+3))) / (6 * STEP ** 2)
    else:
        Y5RAY =lambda i:(-6*F(XRAY(N-3))+24*F(XRAY(N-2))-30*F(XRAY(N-1))+12*F(XRAY(N)))/(6*STEP**2)

for i in range(2,N):
    Y5RAY =lambda i:(F(XRAY(i + 1)) - 2 * F(XRAY(i)) + F(XRAY(i - 1))) / (STEP ** 2)
    razn = Y4RAY(i) - Y5RAY(i) #разность между функциями вычисленными аналитически и численно
    Ekm = numpy.abs(razn) # максимальная погрешность численного дифференцирования
    ek = numpy.sqrt((1 / (N + 1)) * (razn)**2)# среднеквадратичная погрешность численного дифференцирования
print('Ekm=',Ekm,'Ek=',ek)

x = Symbol('x')
y = (x**2)*exp(x) # Функция

# Первая производная
yfirst = y.diff(x)
print(yfirst)

# Вторая производная
ysecond = yfirst.diff(x)
print(ysecond)
test_x = np.linspace(0, 1,100)
# соответствующая y, первая производная, вторая производная
test_y = [y.subs({x:v}) for v in test_x]
test_y_f = [yfirst.subs({x:v}) for v in test_x]
test_y_s = [ysecond.subs({x:v}) for v in test_x]
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(test_x, test_y, color='black', label='function_f')
ax.plot(test_x, test_y_f, color='red', label='first_order')
ax.plot(test_x, test_y_s, color='blue', label='second_order')
ax.plot(test_x, np.zeros(len(test_x)), 'g--', label='y=0')

ax.legend(fontsize=15)
plt.show()


