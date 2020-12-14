#
# https://www.tensorflow.org/tutorials/images/classification
#

print("https://www.tensorflow.org/tutorials/images/classification")

#Imports
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import pathlib
import tensorflow as tf

#Imports of sub-libs from tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))
print("No. of images:")
print(image_count)

print("Opening sample images")

def ShowPic(loc):
#make a simple command to show pictures easily
    pic = PIL.Image.open(loc)
    pic.show()

#display example pictures
#roses
roses = list(data_dir.glob('roses/*'))
ShowPic(str(roses[0]))
ShowPic(str(roses[1]))

#tulips
tulips = list(data_dir.glob('tulips/*'))
ShowPic(str(tulips[0]))
ShowPic(str(tulips[1]))

#Createing the dataset

#Parameters
batch_size = 32
img_height = 180
img_width = 180

#Let's use validation split for developing this model.
#In our case we are going to use 80% of images for training and
#20% for validation.
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split= 0.2,
    subset = "training",
    seed = 123,
    image_size = (img_height, img_width),
    batch_size=batch_size)
print(train_ds)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split= 0.2,
    subset="validation",
    seed = 123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
print(val_ds)

#let's see the class names.
class_names = train_ds.class_names
print(class_names)

#Visualising the data
#lets see the first 9 images from the dataset
x = 9
plt.figure(figsize=(10,10))
for images, labels in train_ds.take(1):
    for i in range(x):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()


for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

#Configure the dataset for performance
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#Standardizing the data
#The RGB channel values are not ideal for a nueral net; 
#in general we should seek to make input valuses small.
#Here, lets standardise the values to become [0,1] range by using
#a rescalling layer  
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

#We will be applying this layer to the database by calling map:
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]

#Now the pixel values should be in [0,1]
#Lets check it just in case
print(np.min(first_image), np.max(first_image))

#Create the model
#Convulation
#if you don't understand what convulation really is, https://en.wikipedia.org/wiki/convolutional_neural_network
num_classes = 5

model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

#Compiling the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#View model summary
modsum = model.summary()
#Training the model

#Let's train the model using these datasets by 
#passing them to model.fit
epochs = 5
history = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = epochs
)

#Visualizing training results
#Let's create plots of loss and accuacy
#on the traing and validation sets.
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show() 

#
#https://www.tensorflow.org/tutorials/images/classification#overfitting
#

#Preventing Overfitting
#Data augmentation
#https://www.tensorflow.org/tutorials/images/classification#data_augmentation
#

data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(img_height, 
                                                              img_width,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)

epochs = 10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

#Visualizing training results
#Let's create plots of loss and accuacy
#on the traing and validation sets.
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show() 

#Visualizing some augmented examples
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
  for i in range(9):
    augmented_images = data_augmentation(images)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy().astype("uint8"))
    plt.axis("off")
plt.show()

#Predicting New Data
file_name = "image"
img_url = input("Path of the image(resolution : 180x180)")
img_path = tf.keras.utils.get_file(file_name, origin=img_url)

img = keras.preprocessing.image.load_img(
      img_path, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} . I tell this with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)