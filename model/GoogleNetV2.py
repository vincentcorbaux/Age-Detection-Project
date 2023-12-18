from keras import Model, Input
from keras.layers import Conv2D, BatchNormalization, Activation, Dropout, Flatten, Dense, MaxPooling2D, concatenate, AveragePooling2D

def conv_block(x, nb_filter, nb_row, nb_col, padding="same", strides=(1, 1), use_bias=False):
    x = Conv2D(nb_filter, (nb_row, nb_col), strides=strides, padding=padding, use_bias=use_bias)(x)
    x = BatchNormalization(axis=-1, momentum=0.9997, scale=False)(x)
    x = Activation("relu")(x)
    return x

def stem(input):
    x = conv_block(input, 32, 3, 3, strides=(2, 2), padding="same")
    x = conv_block(x, 32, 3, 3, padding="same")
    x = conv_block(x, 64, 3, 3)
    x1 = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    x2 = conv_block(x, 96, 3, 3, strides=(2, 2), padding="same")
    x = concatenate([x1, x2], axis=-1)
    x1 = conv_block(x, 64, 1, 1)
    x1 = conv_block(x1, 96, 3, 3, padding="same")
    x2 = conv_block(x, 64, 1, 1)
    x2 = conv_block(x2, 64, 1, 7)
    x2 = conv_block(x2, 64, 7, 1)
    x2 = conv_block(x2, 96, 3, 3, padding="same")
    x = concatenate([x1, x2], axis=-1)
    x1 = conv_block(x, 192, 3, 3, strides=(2, 2), padding="same")
    x2 = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    x = concatenate([x1, x2], axis=-1)
    return x

def inception_A(input):
    a1 = AveragePooling2D((3, 3), strides=(1, 1), padding="same")(input)
    a1 = conv_block(a1, 96, 1, 1)
    a2 = conv_block(input, 96, 1, 1)
    a3 = conv_block(input, 64, 1, 1)
    a4 = conv_block(input, 64, 1, 1)
    a4 = conv_block(a4, 96, 3, 3)
    a4 = conv_block(a4, 96, 3, 3)
    a = concatenate([a1, a2, a3, a4], axis=-1)
    return a

def reduction_A(input):
    ra1 = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(input)
    ra2 = conv_block(input, 384, 3, 3, strides=(2, 2), padding="same")
    ra3 = conv_block(input, 192, 1, 1)
    ra3 = conv_block(ra3, 224, 3, 3)
    ra3 = conv_block(ra3, 256, 3, 3, strides=(2, 2), padding="same")
    ra = concatenate([ra1, ra2, ra3], axis=-1)
    return ra

def inception_base_v4(input):
    net = stem(input)
    for i in range(2):  # Reducing the number of inception A blocks
        net = inception_A(net)
    net = reduction_A(net)
    return net

def inceptionv4(input_shape, dropout_keep, include_top=1):
    inputs = Input(input_shape)
    net_base = inception_base_v4(inputs)
    
    if include_top:
        net = AveragePooling2D((4, 4), padding="valid")(net_base)
        net = Dropout(1 - dropout_keep)(net)
        net = Flatten()(net)
        
        out_age = Dense(units=256, activation="relu", name="age_dense")(net)  # Reducing units in the dense layers
        out_age = Dropout(0.4)(out_age)
        out_age = Dense(units=128, activation="relu", name="age_dense2")(out_age)  # Reducing units in the dense layers
        out_age = Dropout(0.2)(out_age)
        out_age = Dense(units=10, activation="linear", name="age_output")(out_age)  # Output for regression
        
        model = Model(inputs, outputs=[out_age], name="inceptionv4")
        return model
