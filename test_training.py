import matplotlib
from model.ResNet50 import AgeEstimatorModel as ResNetModel
from model.EfficientNetB0 import AgeEstimatorModel1 as EfficientNetModel
import tensorflow as tf
matplotlib.use("Agg")
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import glob

def train_model(model, model_name):
    # compile the model
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
    model.compile(loss="mean_squared_error", optimizer=opt, metrics=["mae"])  # using mean squared error loss

    # train the model
    H = model.fit(aug.flow(trainX, trainY, batch_size=batch_size),
                  validation_data=(testX, testY),
                  steps_per_epoch=len(trainX) // batch_size,
                  epochs=epochs, verbose=1)

    # save the model to disk
    model.save(f"{model_name}_age_detection.model")

    return H

def plot_model_performance(H, model_name):
    # plot training/validation loss/mae
    plt.style.use("ggplot")
    plt.figure()
    N = epochs
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["mae"], label="train_mae")
    plt.plot(np.arange(0, N), H.history["val_mae"], label="val_mae")

    plt.title(f"{model_name} Training Loss and MAE")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/MAE")
    plt.legend(loc="upper right")
    plt.savefig(f"{model_name}_plot.png")


# Paths and parameters initialization
dataset_path = "Faces"  # Update this path based on your dataset location
model_path_resnet = "resnet_age_detection.model"
model_path_effnet = "effnet_age_detection.model"
plot_path_resnet = "resnet_plot.png"
plot_path_effnet = "effnet_plot.png"

# Initial parameters
epochs = 10
lr = 1e-3
batch_size = 64
img_dims = (96, 96, 3)

# Load and preprocess the data
data = []
age_labels = []

# Load image files from the dataset
image_files = [f for f in glob.glob(dataset_path + "/*.jpg", recursive=True)]

random.seed(42)
random.shuffle(image_files)

# Create ground-truth labels from the image paths
for img in image_files:
    image = cv2.imread(img)
    image = cv2.resize(image, (img_dims[0], img_dims[1]))
    image = img_to_array(image)
    data.append(image)

    # Extract age labels from the image paths
    label = int(img.split("/")[-1].split("_")[0])
    age_labels.append(label)

# Pre-process the data
data = np.array(data, dtype="float") / 255.0
labels = np.array(age_labels)

# Split dataset for training and validation
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

# Augmenting dataset
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

# build and train ResNet50 model
model_resnet = ResNetModel.build(width=img_dims[0], height=img_dims[1], depth=img_dims[2], classes=1)
H_resnet = train_model(model_resnet, "ResNet50")
plot_model_performance(H_resnet, "ResNet50")

# build and train EfficientNetB0 model
model_effnet = EfficientNetModel.build(width=img_dims[0], height=img_dims[1], depth=img_dims[2], classes=1)
H_effnet = train_model(model_effnet, "EfficientNetB0")
plot_model_performance(H_effnet, "EfficientNetB0")

# Test the models on a random image
def test_model(model, model_name):
    random_image_path = random.choice(image_files)
    random_image = cv2.imread(random_image_path)
    random_image = cv2.resize(random_image, (img_dims[0], img_dims[1]))

    # Extract the actual age from the image path
    actual_age = int(random_image_path.split("/")[-1].split("_")[0])

    # Preprocess the image for prediction
    random_image = random_image.astype("float") / 255.0
    random_image = img_to_array(random_image)
    random_image = np.expand_dims(random_image, axis=0)

    # Make a prediction using the trained model
    predicted_age = model.predict(random_image)

    # Display the actual age and model's prediction
    print(f"Model: {model_name}")
    print(f"Actual age: {actual_age}")
    print(f"Predicted age: {int(predicted_age[0][0])}\n")


# Testing ResNet50 model
test_model(model_resnet, "ResNet50")

# Testing EfficientNetB0 model
test_model(model_effnet, "EfficientNetB0")
