import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, concatenate, Flatten, Dense


class AgeEstimatorModel2:
    @staticmethod
    def build(width, height, depth, classes):
        model = googlenet((height, width, depth), classes)
        return model
    


# Define the Inception Module
def inception_module(x, filters):
    conv1x1 = Conv2D(filters[0], (1, 1), padding='same', activation='relu')(x)
    
    conv3x3_reduce = Conv2D(filters[1], (1, 1), padding='same', activation='relu')(x)
    conv3x3 = Conv2D(filters[2], (3, 3), padding='same', activation='relu')(conv3x3_reduce)
    
    conv5x5_reduce = Conv2D(filters[3], (1, 1), padding='same', activation='relu')(x)
    conv5x5 = Conv2D(filters[4], (5, 5), padding='same', activation='relu')(conv5x5_reduce)
    
    maxpool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    maxpool_proj = Conv2D(filters[5], (1, 1), padding='same', activation='relu')(maxpool)
    
    inception_output = concatenate([conv1x1, conv3x3, conv5x5, maxpool_proj], axis=-1)
    return inception_output

# Define the GoogLeNet model
def googlenet(input_shape, num_classes):
    input_layer = Input(shape=input_shape)
    
    # Initial Convolution and MaxPooling
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    # Inception Modules
    x = inception_module(x, [64, 64, 128, 32, 32, 32])
    x = inception_module(x, [128, 128, 192, 96, 96, 64])
    
    # Add more inception modules as needed
    
    # Auxiliary Classifier 1
    aux1 = AveragePooling2D((5, 5), strides=(3, 3))(x)
    aux1 = Conv2D(128, (1, 1), padding='same', activation='relu')(aux1)
    aux1 = Flatten()(aux1)
    aux1 = Dense(1024, activation='relu')(aux1)
    aux1 = Dense(num_classes, activation='softmax')(aux1)
    
    # Inception Modules and other layers
    
    # Auxiliary Classifier 2
    aux2 = AveragePooling2D((5, 5), strides=(3, 3))(x)
    aux2 = Conv2D(128, (1, 1), padding='same', activation='relu')(aux2)
    aux2 = Flatten()(aux2)
    aux2 = Dense(1024, activation='relu')(aux2)
    aux2 = Dense(num_classes, activation='softmax')(aux2)
    
    # Main Classifier
    x = AveragePooling2D((7, 7), strides=(1, 1))(x)
    x = Flatten()(x)
    x = Dense(1000, activation='relu')(x)  # Adjust the number of neurons for your specific problem
    x = Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.models.Model(inputs=input_layer, outputs=[x, aux1, aux2])
    
    return model