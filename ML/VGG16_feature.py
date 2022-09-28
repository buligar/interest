from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.utils import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
import numpy as np

# предварительно построенная и обученная модель глубокого обучения VGG16

base_model = VGG16(weights='imagenet', include_top=True)

for i, layer in enumerate(base_model.layers):
    print(i, layer.name, layer.output_shape)

# выделить признаки из слоя block4_pool
model = Model()
img_path = 'cat.jpg'
img = load_img(img_path, target_size=(224, 224))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

feautures = model.predict(x)


def main():
    pass


if __name__ == "__main__":
    main()
