from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np


def main():
    base_model = VGG16(weights='imagenet', include_top=True)
    for i, layer in enumerate(base_model.layers):
        print(i, layer.name, layer.output_shape)


if __name__ == "__main__":
    main()
