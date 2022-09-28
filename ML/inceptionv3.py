from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.optimizers import SGD

# создать базовую предобученную модель
base_model = InceptionV3(weights='imagenet', include_top=False)

# добавить глобальный слой пулинга, выполняющего пространственное усреднение

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(200, activation='softmax')(x)
model = Model(input=base_model.input, output=predictions)

# заморозить все сверточные слои сети InceptionV3

for layer in base_model.layers: layer.trainable = False

# откомпилировать модель (это нужно делать ПОСЛЕ того, как некоторые слои помечены как необучаемые)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

model.fit_generator(...)

# мы решили обучить 2 слоя inception, так что 172 слоя замораживаются, а остальные размораживаются

for layer in model.layers[:172]:
    layer.trainable = False

for layer in model.layers[172:]:
    layer.trainable = True

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# и снова обучаем модель (на этот раз настраеваем 2 верхних слоя inception) и верхние слои Dense

model.fit_generator(...)


def main():
    pass


if __name__ == "__main__":
    main()
