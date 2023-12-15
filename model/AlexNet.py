import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D,BatchNormalization, GlobalAveragePooling2D

class AlexNetAgeDetection:
    @staticmethod
    def build(width, height, depth, classes):
        base_model = Sequential()

        # 1st Convolutional Layer
        base_model.add(Conv2D(filters=96, input_shape=(width,height,depth), kernel_size=(11,11),
        strides=(4,4), padding='valid'))
        base_model.add(Activation('relu'))
        # Pooling 
        base_model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
        # Batch Normalisation before passing it to the next layer
        base_model.add(BatchNormalization())

        # 2nd Convolutional Layer
        base_model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='valid'))
        base_model.add(Activation('relu'))
        # Pooling
        base_model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        # Batch Normalisation
        base_model.add(BatchNormalization())

        # 3rd Convolutional Layer
        base_model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
        base_model.add(Activation('relu'))
        # Batch Normalisation
        base_model.add(BatchNormalization())

        # 4th Convolutional Layer
        base_model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
        base_model.add(Activation('relu'))
        # Batch Normalisation
        base_model.add(BatchNormalization())

        # 5th Convolutional Layer
        base_model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
        base_model.add(Activation('relu'))
        # Pooling
        base_model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        # Batch Normalisation
        base_model.add(BatchNormalization())

        # Passing it to a dense layer
        base_model.add(Flatten())

        # 1st Dense Layer
        base_model.add(Dense(4096, input_shape=(width * height * depth,)))
        base_model.add(Activation('relu'))
        # Add Dropout to prevent overfitting
        base_model.add(Dropout(0.4))
        # Batch Normalisation
        base_model.add(BatchNormalization())

        # 2nd Dense Layer
        base_model.add(Dense(4096))
        base_model.add(Activation('relu'))
        # Add Dropout
        base_model.add(Dropout(0.4))
        # Batch Normalisation
        base_model.add(BatchNormalization())

        # 3rd Dense Layer
        base_model.add(Dense(1000))
        base_model.add(Activation('relu'))
        # Add Dropout
        base_model.add(Dropout(0.4))
        # Batch Normalisation
        base_model.add(BatchNormalization())

        # Output Layer
        base_model.add(Dense(1))  # Assuming 17 output classes for age prediction
        base_model.add(Activation('softmax'))
    
        return base_model