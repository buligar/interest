import numpy
import numpy as np
import matplotlib.pyplot as plt

m = int(input())
k = int(input())
j = int(input())
n = int(input())
if 1 > m or 4 < m:
    print('Введите 1<m<4')
elif 1 > k or 4 < k:
    print('Введите 1<k<4')
elif 1 > j or 4 < j:
    print('Введите 1<j<4')
elif 50 > n or 1000 < n:
    print('Введите 50<m<1000')
else:
    pi = 3.141592
    y = lambda x: np.sin((pi * x ** m) / 2) ** k - ((1 - x) ** j)
    fig = plt.subplots()
    x = np.linspace(0, 1, n)
    plt.plot(x, y(x))
    plt.show()
    maxim=numpy.argmax(y(x))
    minim = numpy.argmin(y(x))
    mean = numpy.mean(y(x))
    aver = numpy.average(y(x))
    aver2 = numpy.square(aver)
    sr_znac = numpy.sqrt(abs(mean))
    std = numpy.std(y(x))
    print('max=',maxim,'min=',minim,'Среднее знач=',mean,'Средний квадрат=',aver2,'Среднеквадратическое значение=',sr_znac,'Среднеквадратичное отклонение от среднего значения=',std)

for i in range(1001):
    print(i)