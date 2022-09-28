from keras.optimizers import SGD
from keras.applications import VGG16
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

os.environ["QT_QPA_PLATFORM"] = "wayland"
# готовая модель с предобученными на наборе imagenet весами

model = VGG16(weights='imagenet', include_top=True)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')

# сделать размер таким же, как у изображений, на которых обучалась модель

im = cv2.resize(cv2.imread('cat.jpg'), (224, 224))
im = np.expand_dims(im, axis=0)

# предсказание
out = model.predict(im)
plt.plot(out.ravel())
plt.show()

print(np.argmax(out))


def main():
    pass


if __name__ == "__main__":
    main()
