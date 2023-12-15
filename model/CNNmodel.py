import keras
from keras.models import Sequential, Model
from keras.layers import Dense,Conv2D,AveragePooling2D,GlobalAveragePooling2D, Dropout
import numpy as np

class AgeEstimatorModel5:
    @staticmethod
    def build(width, height, depth, classes):
        # Defining the architecture of the sequential neural network.
        base_model = Sequential()
        # Input layer with 32 filters, followed by an AveragePooling2D layer.
        base_model.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(height, width, depth)))    # 3rd dim = 1 for grayscale images.
        base_model.add(AveragePooling2D(pool_size=(2,2)))
        # Three Conv2D layers with filters increasing by a factor of 2 for every successive Conv2D layer.
        base_model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
        base_model.add(AveragePooling2D(pool_size=(2,2)))
        base_model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
        base_model.add(AveragePooling2D(pool_size=(2,2)))
        base_model.add(Conv2D(filters=256, kernel_size=3, activation='relu'))
        base_model.add(AveragePooling2D(pool_size=(2,2)))
        # A GlobalAveragePooling2D layer before going into Dense layers below.
        # GlobalAveragePooling2D layer gives no. of outputs equal to no. of filters in last Conv2D layer above (256).
        base_model.add(GlobalAveragePooling2D())
        # One Dense layer with 132 nodes so as to taper down the no. of nodes from no. of outputs of GlobalAveragePooling2D layer above towards no. of nodes in output layer below (7).
        base_model.add(Dense(132, activation='relu'))
        # Output layer with 7 nodes (equal to the no. of classes).
        base_model.add(Dense(classes, activation='softmax'))
        return base_model
