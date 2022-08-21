import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.utils
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense,Input,Conv2D,MaxPooling2D,Flatten
from tensorflow.keras.datasets import mnist,cifar10
from tensorflow.keras.utils import to_categorical

from skimage.feature import greycomatrix, greycoprops
from skimage import io

from PyQt5 import QtCore, QtWidgets, uic

tf.random.set_seed(1)

# tf.constant(value,dtype=None,shape=None,name="Const")
# value - значение тензора
# dtype - тип данных тензора
# shape - размерность тензора
# name - имя тензора

### Пример 1 ###

# x = tf.Variable([[2.0]])
# y = tf.Variable([[-4.0]])
# with tf.GradientTape() as tape:
#     f = (x+y)**2+2*x*y
#
# df = tape.gradient(f,[x,y])
# print(df[0],df[1], sep="\n")
# Пример 2
# x=tf.Variable(-1.0)
# y=lambda: x**2-x
# N=100
# opt = tf.optimizers.SGD(learning_rate=0.1)
# for n in range(N):
#     opt.mininize(y,[x])
# print(x.numpy())

### Пример 3 ###

# #создание тензоров с статическими параметрами
# a=tf.constant(1,shape=(1,1)) # две оси
# b=tf.constant([1,2,3,4]) # одна ось+матрица
# c=tf.constant([[1,2],
#                [3,4],
#                [5,6]], dtype=tf.float16)# две оси+двумерная матрица
# a2=tf.cast(a,dtype=tf.float32) #изменить тип данных тензора
# b1=np.array(b) #изменить тензор в список(массив) numpy
# b2=b.numpy() #изменить тензор в список(массив) numpy вариант2
# print(a,a2, sep="\n")
# print(b,b1,b2, sep="\n\n")
# print(c)
# #создание тензоров с изменяющимися параметрами
# v1=tf.Variable(-1.2)
# v2=tf.Variable([4,5,6,7],dtype=tf.float32)
# v3=tf.Variable(b)
# print(v1,v2,v3, sep="\n\n")
# #изменение тензоров
# v1.assign(0) #замена значений тензора
# v2.assign([0,1,6,7])
# v3.assign_add([1,1,1,1]) # Прибавление значений к существующему тензору
# v1.assign_sub(5) # Вычитание значений из тензора
# print(v1,v2,v3, sep="\n\n")
#
# val_0=v3[0] # первый элемент
# val_12=v3[1:3] # элементы со 2-го по 3-ий
# val_0.assign(10) # прибавлят к тензору число(10), но статический тензор остается неизменным
# print(v3,val_0,val_12,sep="\n")
#
# x=tf.constant(range(10))+5 # добавление чисел до нужного массива
# x_indx=tf.gather(x,[0,4]) # выбор элементов массива
# print(x,x_indx,sep="\n")
#
# v2=tf.constant([[1,2,7],[3,4,8],[5,6,9]]) # массив
# val_indx=v2[(1,2)] # выбор элементов массива
# print(val_indx)
#
# v2=tf.constant([[1,2,7],[3,4,8],[5,6,9]]) # массив
# val_indx=v2[:2,-1] # выбор элементов массива start:stop:step
# print(val_indx)

# a=tf.constant(range(30))
# b=tf.reshape(a,[5,6]) # превращение в матрицу
# b_T=tf.transpose(b,perm=[1,0]) #транспонирование матрицы(поменять строки и столбы местами)
# print(b_T)

### Пример 4 ###

# Формирование тензора
# a=tf.zeros((3,3)) #создает тензор из нулей
# b=tf.ones((4,3)) # создает тензор из единиц
# c=tf.zeros_like(a) #заполние нулями выбранного тензора
# d=tf.eye(3) #создание тензора где по главной диагонали единицы, а в остальном нули
# e=tf.identity(c) # тот же выбранный тензор, но с нулями
# f=tf.fill((2,3),-1) #формирование тензора с заданными размерностями и значениями
# g=tf.range(1,11,0.2) #создание тензора с заданным интервалом
# print(a,b,c,d,e,f,g)

# генерация случайных чисел
# a=tf.random.normal((2,4),0,0.1)
# b=tf.random.uniform((2,2),-1,1)
# c=tf.random.set_seed(1) # сохрание тензора со случайными числами,который был сгенерировани ранее
# d=tf.random.truncated_normal((1,5),-1,0.1)
# print(a,b,c,d,sep="\n")

# математические операции
# a=tf.constant([1,2,3])
# b=tf.constant([9,8,7])
# c=tf.add(a,b) # сложение
# d=tf.subtract(a,b) #вычитание
# e=tf.divide(a,b) #деление
# f=tf.multiply(a,b) #умножение
# g=tf.tensordot(a,b,axes=0) #внешнее векторное  умножение(построение матрицы из векторов)
# h=tf.tensordot(a,b,axes=1) #внутреннее векторное умножение (получение числа матрицы)
# print(c,d,e,f,g,h,sep="\n")

# a2=tf.constant(tf.range(1,10),shape=(3,3))
# b2=tf.constant(tf.range(5,14),shape=(3,3))
# c=tf.matmul(a2,b2) #матричное умножение по-другому (a2@b2)
# print(c)

# a = tf.constant([1, 2, 3])
# b = tf.constant([9, 8, 7])
# m = tf.tensordot(a, b, axes=0)
# sum = tf.reduce_sum(m)  # сумма всех элементов матрицы (144)
# sum1 = tf.reduce_sum(m, axis=0)  # сумма по столбцам
# sum2 = tf.reduce_sum(m, axis=1)  # сумма по строкам
# mean = tf.reduce_mean(m)  # среднеарифметическое (144/9)
# max = tf.reduce_max(m)  # макс. элем. матрицы(27)
# min = tf.reduce_min(m, axis=1)  # мин. элем. матрицы по столбцам
# prod = tf.reduce_prod(m, axis=0)  # произведения элементов матрицы по столбцам
# sqrt = tf.sqrt(tf.cast(a, dtype=tf.float32))  # корень из вектора c преобразованием в вещественный тип
# square = tf.square(a)  # квадрат из вектора
# sin = tf.sin(tf.range(-3.14, 3.14, 1))  # синус для углов
# cos = tf.cos(tf.range(-3.14, 3.14, 1))  # косинус для углов
# print(m, sum, sum1, sum2, mean, max, min, prod, sqrt, square, sin, cos, sep="\n")

### Пример 5 ###

# w=tf.Variable(tf.random.normal((3,2))) #Гауссовские нормальные случайные величины
# b=tf.Variable(tf.zeros(2,dtype=tf.float32))
# x=tf.Variable([[-2.0,1.0,3.0]])
# with tf.GradientTape() as tape:
#     y=x@w+b # матричное умножение и прибавление b
#     loss=tf.reduce_mean(y**2)
# df=tape.gradient(loss,[w,b])
# print(df[0],df[1],sep="\n")

### Пример 6 ###

# x=tf.Variable(0,dtype=tf.float32)
# b=tf.constant(1.5)
#
# with tf.GradientTape() as tape:
#     f=(x+b)**2+2*b
# df = tape.gradient(f,[x,b])
# print(df[0],df[1],sep="\n")

### Пример 7 ###
# x=tf.Variable(0,dtype=tf.float32)
# b=tf.Variable(1.5)
# with tf.GradientTape(watch_accessed_variables=False) as tape:
#     tape.watch(x)
#     y=2*x
#     f=y*y
# df=tape.gradient(f,[x,y])
# print(df)

### Пример 8 ###
# x=tf.Variable(1.0)
# with tf.GradientTape() as tape:
#     y=[2.0,3.0]*x**2
# df=tape.gradient(y,x)
# print(df)
### Пример 9 ###
# x=tf.Variable(1.0)
# with tf.GradientTape() as tape:
#     if x<2.0:
#         y=tf.reduce_sum([2.0,3.0]*x**2)
#     else:
#         y=x**2
# df=tape.gradient(y,x)
# print(df)

### Пример 10 частые ошибки ###
# x=tf.Variable(1.0)
# y=2*x+1 #так делать нельзя
# with tf.GradientTape() as tape:
#     #y=2*x+1 должна здесь стоять
#     z=y**2
# df=tape.gradient(z,x)
# print(df)

# x=tf.Variable(1.0)
# for n in range(2):
#     with tf.GradientTape() as tape:
#         y=x**2+2*x
#     df = tape.gradient(y,x)
#     print(df)
#     #x=x+1 # так делать нельзя
#     x.assign_add(1.0)

# x=tf.Variable(1.0)
# with tf.GradientTape() as tape:
#     y=tf.constant(2.0)+np.square(x) # нельзя использовать посторонние пакеты
# df=tape.gradient(y,x)
# print(df)

# x=tf.Variable(1) #нельзя с целочисленными
# with tf.GradientTape() as tape:
#     y=x*x
# df=tape.gradient(y,x)
# print(df)

# x=tf.Variable(1.0)
# w=tf.Variable(2.0)
#
# with tf.GradientTape() as  tape:
#     s=w+x
#     # w.assign_add(x) # теряется связь
#     y=s**2
# df=tape.gradient(y,x)
# print(df)

### Оптимизация ###
# TOTAL_POINTS=1000 # кол-во точек
# x=tf.random.uniform(shape=[TOTAL_POINTS], minval=0,maxval=10) # Генерация вектора
# noise=tf.random.normal(shape=[TOTAL_POINTS],stddev=0.2) #Генерация точек (шум)
# k_true=0.7
# b_true=2.0
# y=x*k_true+b_true+noise # линейная функция
# # plt.scatter(x,y,s=2)
# # plt.show()
# k=tf.Variable(0.0)
# b=tf.Variable(0.0)
# EPOCHS=50 # кол-во итераций градиентного спуска
# learning_rate=0.02 # параметр шага обучения
# BATCH_SIZE = 100 # 100 частей
# num_steps=TOTAL_POINTS // BATCH_SIZE
# # opt=tf.optimizers.SGD(momentum=0.5,nesterov=True,learning_rate=0.02) #№2 Стохастический градиент и метод моментов и метод Нестерова
# # opt=tf.optimizers.Adagrad(learning_rate=0.2) # метод Adagrad
# # opt=tf.optimizers.Adadelta(learning_rate=4.0) # метод Adadelta
# # opt=tf.optimizers.RMSprop(learning_rate=0.01)
# opt=tf.optimizers.Adam(learning_rate=0.1) #№1 модификация метода Adagrad
# for n in range(EPOCHS):
#     for n_batch in range(num_steps):
#         y_batch=y[n_batch*BATCH_SIZE:(n_batch+1)*BATCH_SIZE]
#         x_batch=x[n_batch*BATCH_SIZE:(n_batch+1)*BATCH_SIZE]
#         with tf.GradientTape() as tape: # вычисление частных производных
#             f=k*x_batch+b
#             loss=tf.reduce_mean(tf.square(y_batch-f))
#         dk,db=tape.gradient(loss,[k,b])
#         # k.assign_sub(learning_rate*dk)
#         # b.assign_sub(learning_rate*db)
#         opt.apply_gradients(zip([dk,db],[k,b]))
# print(k,b,sep="\n")
# y_pr=k*x+b
# plt.scatter(x,y,s=2)
# plt.scatter(x,y_pr,c='r',s=2)
# plt.show()

### Построение модели ###
# class DenseNN(tf.Module):
#     def __init__(self,outputs):
#         super().__init__()
#         self.outputs=outputs
#         self.fl_init=False
#     def __call__(self,x):
#         if not self.fl_init:
#             self.w=tf.random.truncated_normal((x.shape[-1],self.outputs),stddev=0.1,name="w")
#             self.b=tf.zeros([self.outputs],dtype=tf.float32,name="b")
#
#             self.w=tf.Variable(self.w)
#             self.b=tf.Variable(self.b)
#             self.fl_init=True
#         y=x@self.w+self.b
#         return y
#
# model=DenseNN(1)
# x_train=tf.random.uniform(minval=0,maxval=10,shape=(100,2)) # сформировали обучающую выборку
# y_train=[a+b for a,b in x_train]
#
# loss=lambda x,y:tf.reduce_mean(tf.square(x-y)) # сформировали функцию потерь
# opt=tf.optimizers.Adam(learning_rate=0.01) # сформировали оптимизатор
#
# EPOCHS=50
# for n in range(EPOCHS):
#     for x,y in zip(x_train,y_train):
#         x=tf.expand_dims(x,axis=0)
#         y=tf.constant(y,shape=(1,1))
#         with tf.GradientTape() as tape:
#             f_loss=loss(y,model(x))
#         grads=tape.gradient(f_loss,model.trainable_variables)
#         opt.apply_gradients(zip(grads,model.trainable_variables))
#     print(f_loss.numpy())
# print(model.trainable_variables)
# print(model(tf.constant([[1.0,2.0]])))

### Нейросеть для распознавания изображений цифр ###

# class DenseNN(tf.Module):
#     def __init__(self, outputs, activate="relu"):
#         super().__init__()
#         self.outputs = outputs
#         self.activate = activate
#         self.fl_init = False
#
#     def __call__(self, x):
#         if not self.fl_init:
#             self.w = tf.random.truncated_normal((x.shape[-1], self.outputs), stddev=0.1, name="w")
#             self.b = tf.zeros([self.outputs], dtype=tf.float32, name="b")
#
#             self.w = tf.Variable(self.w)
#             self.b = tf.Variable(self.b, trainable=False)
#
#             self.fl_init = True
#
#         y = x @ self.w + self.b
#
#         if self.activate == "relu":
#             return tf.nn.relu(y)
#         elif self.activate == "softmax":
#             return tf.nn.softmax(y)
#
#         return y
#
#
# class SequentialModule(tf.Module):
#     def __init__(self):
#         super().__init__()
#         self.layer_1 = DenseNN(128)
#         self.layer_2 = DenseNN(10, activate="softmax")
#
#     def __call__(self, x):
#         return self.layer_2(self.layer_1(x))
#
#
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# x_train = x_train / 255
# x_test = x_test / 255
#
# x_train = tf.reshape(tf.cast(x_train, tf.float32), [-1, 28*28])
# x_test = tf.reshape(tf.cast(x_test, tf.float32), [-1, 28*28])
#
# y_train = to_categorical(y_train, 10)
#
#
# model = SequentialModule()
# # layer_1 = DenseNN(128)
# # layer_2 = DenseNN(10, activate="softmax")
# #print(model.submodules)
#
# cross_entropy = lambda y_true, y_pred: tf.reduce_mean(tf.losses.categorical_crossentropy(y_true, y_pred))
# opt = tf.optimizers.Adam(learning_rate=0.001)
#
# BATCH_SIZE = 32
# EPOCHS = 10
# TOTAL = x_train.shape[0]
#
# train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)
#
#
# @tf.function # преобразование в графовое представление
# def train_batch(x_batch, y_batch):
#     with tf.GradientTape() as tape:
#         f_loss = cross_entropy(y_batch, model(x_batch))
#
#     grads = tape.gradient(f_loss, model.trainable_variables)
#     opt.apply_gradients(zip(grads, model.trainable_variables))
#
#     return f_loss
#
#
# for n in range(EPOCHS):
#     loss = 0
#     for x_batch, y_batch in train_dataset:
#         loss += train_batch(x_batch, y_batch)
#
#     print(loss.numpy())
#
#
# y = model(x_test)
# y2 = tf.argmax(y, axis=1).numpy()
# acc = len(y_test[y_test == y2])/y_test.shape[0] * 100
# print(acc)
#
# acc = tf.metrics.Accuracy()
# acc.update_state(y_test, y2)
# print( acc.result().numpy() * 100 )

### Keras ###

# class DenseLayer(tf.keras.layers.Layer):
#     def __init__(self, units=1): #кол-во нейронов
#         super().__init__() # конструктор базового класса
#         self.units = units # хранит кол-во нейронов
#         self.rate = 0.01
#
#     def build(self, input_shape): # для инициализации переменных
#         self.w = self.add_weight(shape=(input_shape[-1], self.units),
#                                  initializer="random_normal",
#                                  trainable=True)
#         self.b = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True)
#
#     def call(self, inputs):
#         regular = tf.reduce_mean(tf.square(self.w))
#         self.add_loss(regular) # функция потерь
#         self.add_metric(regular, name="mean square weights")
#
#         return tf.matmul(inputs, self.w) + self.b
#
#
# class NeuralNetwork(tf.keras.Model):
#     def __init__(self):
#         super().__init__()
#         self.layer_1 = DenseLayer(128) # 128 нейронов
#         self.layer_2 = DenseLayer(10) # 10 выходных нейронов
#
#     def call(self, inputs): # как обрабатывать входной слой
#         x = self.layer_1(inputs)
#         x = tf.nn.relu(x)
#         x = self.layer_2(x)
#         x = tf.nn.softmax(x)
#         return x
#
#
# model = NeuralNetwork()
#
# # model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),
# #              loss=tf.losses.categorical_crossentropy,
# #              metrics=['accuracy'])
#
# model.compile(optimizer='adam',
#              loss='categorical_crossentropy',
#              metrics=['accuracy'])
#
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# x_train = x_train / 255
# x_test = x_test / 255
#
# x_train = tf.reshape(tf.cast(x_train, tf.float32), [-1, 28*28])
# x_test = tf.reshape(tf.cast(x_test, tf.float32), [-1, 28*28])
#
# y_train = to_categorical(y_train, 10)
# y_test_cat = to_categorical(y_test, 10)
#
# model.fit(x_train, y_train, batch_size=32, epochs=5)
#
# print( model.evaluate(x_test, y_test_cat) )

### Настройка модели через fit() ###
# (x_train,y_train),(x_test,y_test)=mnist.load_data() # загрузка
#
# x_train=x_train.reshape(-1,784)/255.0 # стандартизация
# x_test=x_test.reshape(-1,784)/255.0
#
# # сделать акцент на единице
# sample_weight=np.ones(shape=(len(x_train),))
# sample_weight[y_train==1]=5.0
#
# y_train=keras.utils.to_categorical(y_train,10) #Приведение значений к onehot vector
# y_test=keras.utils.to_categorical(y_test,10)
#
# validation_split=0.2
# validation_split_index=np.ceil(x_train.shape[0]*validation_split).astype('int32') #Ceil-округление до наименьшего целого
# # выборка валидации
# x_train_val=x_train[:validation_split_index]
# y_train_val=y_train[:validation_split_index]
# # обучающая выборка
# x_train_data=x_train[validation_split_index:]
# y_train_data=y_train[validation_split_index:]
#
# #обучающая выборка
# train_dataset=tf.data.Dataset.from_tensor_slices((x_train_data,y_train_data)) #перемешанные значения
# train_dataset=train_dataset.shuffle(buffer_size=1024).batch(64) # разбиение на батчи
#
# # валидационная выборка
# val_dataset=tf.data.Dataset.from_tensor_slices((x_train_val,y_train_val))
# val_dataset=val_dataset.batch(64)
#
# model=keras.Sequential([  #Создание модели
#     layers.Input(shape=(784,)),
#     layers.Dense(128,activation='relu'),
#     layers.Dense(64,activation='relu'),
#     layers.Dense(10,activation='softmax'),
# ])
#
# model.compile(optimizer='adam', # оптимизатор
#               loss='categorical_crossentropy',#функция потерь
#               metrics=['accuracy']) # метрика точность
#
# # регулировка весов
# # class_weight={
# #     0:1000.0,
# #     1:1.0,
# #     2:1.0,
# #     3:1.0,
# #     4:1.0,
# #     5:1.0,
# #     6:1.0,
# #     7:1.0,
# #     8:1.0,
# #     9:1.0,
# # }
#
# # Пример генератора последовательностей
# # class DigitsLimit(keras.utils.Sequence):
# #     def __init__(self,x,y,batch_size,max_len=-1): # Конструктор
# #         self.batch_size=batch_size # Сохраняем batch_size
# #         self.x=x[:max_len] # Сохраняем x
# #         self.y=y[:max_len] # Сохраняем y
# #     def __len__(self): # возвращает число минибатчей выборки
# #         return int(np.ceil(self.x.shape[0]/self.batch_size))
# #     def __getitem__(self, idx): # возвращает текущий минибатч
# #         batch_x=self.x[idx*self.batch_size:(idx+1)*self.batch_size]
# #         batch_y=self.y[idx*self.batch_size:(idx+1)*self.batch_size]
# #         return batch_x,batch_y
# #     def on_epoch_end(self):
# #         p=np.random.permutation(len(self.x))
# #         self.x=self.x[p]
# #         self.y=self.y[p]
# #         print("on_epoch_end")
# # sequence=DigitsLimit(x_train,y_train,64,10000)
#
# # вычисляет значение потерь после обработки очередного батча
# class CustomCallback(keras.callbacks.Callback):
#     def on_train_begin(self,logs): # начало обучения
#         self.per_batch_losses=[] # создаем коллекцию
#     def on_batch_end(self,batch,logs): # конец обработки батча
#         self.per_batch_losses.append(logs.get("loss")) # сохраняем потери в коллекцию
#     def on_train_end(self,logs): # конец обучения
#         print(self.per_batch_losses[:5]) # выводим первые 5 значений коллекции
#
# #Досроная остановка
# callbacks=[
#     keras.callbacks.EarlyStopping(
#         monitor="loss",
#         min_delta=0.01, # Изменение в 0.01
#         patience=3, # на протяжении трех эпох
#         verbose=1),
#     keras.callbacks.ModelCheckpoint(
#         filepath="mymodel_{epoch}", # Путь к папке с данными, epoch-значение текущей эпохи
#         save_best_only=True, # Производить сохранения если показатель качества улучшился
#         monitor="loss", # какой показатель качества отслеживать
#         verbose=1),
#     CustomCallback(),
# ]
# # настройки модели
# history=model.fit(
#     x_train,y_train, # вариант 1 выборка
#     # train_dataset, # вариант 2 выборка
#     # sequence, # вариант 3 выборка
#     epochs=5, # кол-во эпох
#     # validation_split=0.2, # валидационная выборка
#     # shuffle=True, # Перемешивание True-вкл, False-выкл
#     callbacks=callbacks # досрочная остановка
# # Доп параметры
#           # validation_data=val_dataset,
#           # steps_per_epoch=100, # 500 минибатчей
#           # validation_steps=5, #берется первые 5 минибатчей
#           # class_weight=class_weight, # регулировка весов класса
#           # sample_weight=sample_weight # регилировка отдельного веса образца
#     )
# print(history.history) # вывод значений
#
# # model=keras.models.load_model('mymodel_3') # загрузка сохраненной модели
# # print(model.evaluate(x_test,y_test)) # evaluate - качество работы модели

### Настройка модели через compile() ###

# (x_train,y_train),(x_test,y_test)=mnist.load_data()
#
# # x_train=x_train.reshape(-1,784)/255.0
# # x_test=x_test.reshape(-1,784)/255.0
# x_train=x_train/255.0
# x_test=x_test/255.0
#
# y_train=keras.utils.to_categorical(y_train,10)
# y_test=keras.utils.to_categorical(y_test,10)
#
# enc_input=layers.Input(shape=(28,28,1)) # Подаем изображение
# x=layers.Conv2D(32,3,activation='relu')(enc_input)
# x=layers.MaxPooling2D(2,padding='same')(x)
# x=layers.Conv2D(64,3,activation='relu')(x)
# x=layers.MaxPooling2D(2,padding='same')(x)
# x=layers.Flatten()(x)
# hidden_output=layers.Dense(8,activation='linear')(x) # вектор скрытого состояния
# #декодер(Восстановление изображения)
# x=layers.Dense(7*7*8,activation='relu')(hidden_output)
# x=layers.Reshape((7,7,8))(x)
# x=layers.Conv2DTranspose(64,5,strides=(2,2),activation="relu",padding='same')(x)
# x=layers.BatchNormalization()(x)
# x=layers.Conv2DTranspose(32,5,strides=(2,2),activation="linear",padding='same')(x)
# x=layers.BatchNormalization()(x)
# dec_output=layers.Conv2DTranspose(1,3,activation="sigmoid",padding='same',name="dec_output")(x)
# # Классификатор
# x2=layers.Dense(128,activation='relu')(hidden_output)
# class_output=layers.Dense(10,activation='softmax',name="class_output")(x2)
#
# model=keras.Model(enc_input,[dec_output,class_output])
#
# #Простая модель
# # model=keras.Sequential([
# #     layers.Input(shape=(784,)),
# #     layers.Dense(128,activation='relu'),
# #     layers.Dense(64,activation='relu'),
# #     layers.Dense(10,activation='softmax'),
# # ])
#
# # Собственная функция потерь
# # def myloss(y_true,y_pred):
# #     return tf.reduce_mean(tf.square(y_true-y_pred))
#
# # 2 Собственная функция потерь
# # class MyLoss(keras.losses.Loss):
# #     def __init__(self,alpha=1.0,beta=1.0):
# #         super(MyLoss,self).__init__()
# #         self.alpha=alpha
# #         self.beta=beta
# #     def call(self,y_true,y_pred): # y_true-требуемое значение y_pred-значение, которое получается
# #         return tf.reduce_mean(tf.square(self.alpha*y_true-self.beta*y_pred))
# # # Создание своих метрик
# # class CategoricalTruePositives(keras.metrics.Metric):
# #     def __init__(self,name="my_metric"): # Конструктор
# #         super().__init__(name=name) # базовый класс
# #         self.true_positives=self.add_weight(name="acc",initializer="zeros") # тензор
# #         self.count=tf.Variable(0.0) # тензор
# #     def update_state(self,y_true,y_pred,sample_weight=None):
# #         y_pred=tf.reshape(tf.argmax(y_pred,axis=1),shape=(-1,1))
# #         y_true=tf.reshape(tf.argmax(y_true,axis=1),shape=(-1,1))
# #         values=tf.cast(y_true,"int32")==tf.cast(y_pred,"int32")
# #         if sample_weight is not None:
# #             sample_weight=tf.cast(sample_weight,"float32")
# #             values=tf.multiply(values,sample_weight)
# #         values=tf.cast(values,"float32")
# #         self.true_positives.assign_add(tf.reduce_mean(values)) #вычисляем долю правильно классифицированных изображений в пределах одного мини батча
# #         self.count.assign_add(1.0) # подсчет мини-батчей
# #     def result(self):
# #         return self.true_positives/self.count
# #     def reset_state(self): # обнуление состояния
# #         self.true_positives.assign(0.0)
# #         self.count.assign(0.0)
#
# model.compile(
#     optimizer=keras.optimizers.Adam(learning_rate=0.01),
#     # loss=keras.losses.CategoricalCrossentropy(),
#     # loss=myloss # Собственная функция потерь
#     # loss=MyLoss(0.5,0.5), # 2 Собственная функция потерь
#     # loss=['mean_squared_error','categorical_crossentropy'], #3 функция потерь для изображения и классификации
#     loss={
#         'dec_output':'mean_squared_error',
#         'class_output':'categorical_crossentropy'
#     },
#     loss_weights=[1.0,0.5],
#     metrics={
#         'dec_output':None,
#         'class_output':'acc'
#     }
#     # metrics=[keras.metrics.CategoricalAccuracy()]
#     # metrics=[keras.metrics.CategoricalAccuracy(),CategoricalTruePositives()],
#     )
#
# # model.fit(x_train,y_train,epochs=5) # Стандарт
# # model.fit(x_train,[x_train,y_train],epochs=1) # ver 2
# model.fit(x_train,{'dec_output':x_train,'class_output':y_train},epochs=1)
#
# p=model.predict(tf.expand_dims(x_test[0],axis=0)) # Проверка тестового изображения
#
# print(tf.argmax(p[1],axis=1).numpy())
#
# plt.subplot(121)
# plt.imshow(x_test[0],cmap='gray')
# plt.subplot(122)
# plt.imshow(p[0].squeeze(),cmap='gray')
# plt.show()

### Способы сохранения и загрузки моделей ###

# (x_train,y_train),(x_test,y_test)=mnist.load_data()
#
# x_train=x_train.reshape(-1,784)/255.0
# x_test=x_test.reshape(-1,784)/255.0
#
# y_train=keras.utils.to_categorical(y_train,10)
# y_test=keras.utils.to_categorical(y_test,10)
#
# # Простая модель
# # model=keras.Sequential([
# #     layers.Dense(128,activation='relu'),
# #     layers.Dense(10,activation='softmax')
# # ])
# class NeuralNetwork(tf.keras.Model):
#     def __init__(self,units): # units- кол-во нейронов на каждом слое
#         super().__init__()
#         self.units=units
#         self.model_layers=[layers.Dense(n,activation='relu') for n in self.units]
#     def call(self,inputs): # inputs - входной сигнал
#         x=inputs
#         for layer in self.model_layers:
#             x=layer(x)
#         return x # формирует выходной тензор
#     def get_config(self):
#         return {'units':self.units}
#     @classmethod
#     def from_config(cls,config):
#         return cls(**config)
# class NeuralNetworkLinear(tf.keras.Model):
#     def __init__(self,units): # units- кол-во нейронов на каждом слое
#         super().__init__()
#         self.units=units
#         self.model_layers=[layers.Dense(n,activation='linear') for n in self.units]
#     def call(self,inputs): # inputs - входной сигнал
#         x=inputs
#         for layer in self.model_layers:
#             x=layer(x)
#         return x # формирует выходной тензор
#     def get_config(self):
#         return {'units':self.units}
#     @classmethod
#     def from_config(cls,config):
#         return cls(**config)
#
# model=NeuralNetwork([128,10])
# model2=NeuralNetwork([128,10])
#
# y=model.predict(tf.expand_dims(x_test[0],axis=0))
# print(y)
# y=model2.predict(tf.expand_dims(x_test[0],axis=0))
# print(y)
#
# # # model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
# # # model.fit(x_train,y_train,epochs=5)
# #
# # model.save('16_model') # Сохраняем все характеристики модели в путь 16_model
# # model_loaded=keras.models.load_model('16_model',custom_objects={"NeuralNetwork":NeuralNetworkLinear}) # из пути восстанавливаем модель
# #
# # y=model_loaded.predict(tf.expand_dims(x_test[0],axis=0))
# # print(y)
# # # model_loaded.evaluate(x_test,y_test)
#
# #считываем и записываем веса только после пропускания через модели входного сигнала
# #иначе возникнет ошибка из-за отсутсвия начальной инициализации весов
# # weights=model.get_weights() # Получить вес коэф. из 1 модели
# # model2.set_weights(weights) # установить вес коэф для 2 модели
#
# model.save_weights('model_weights.h5') # Указываем путь куда сохранять
# model2.load_weights('model_weights.h5') # считываем вес коэф
# y=model2.predict(tf.expand_dims(x_test[0],axis=0))
# print(y)



