from keras.applications import EfficientNetB0
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model


class AgeEstimatorModel1:
    @staticmethod
    def build(width, height, depth, classes):
        # Load EfficientNetB0 with pre-trained weights, without the top classification layers
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(height, width, depth))

        # Add custom layers for age regression
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(classes, activation='linear')(x)  # Linear activation for regression

        # Define the model
        model = Model(inputs=base_model.input, outputs=predictions)

        return model
