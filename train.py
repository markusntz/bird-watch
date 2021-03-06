#%%
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator, img_to_array
from keras import backend as K
from keras.utils import to_categorical

import glob
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import random 


class LeNet:

    @staticmethod
    def build_model(width, height, depth):
    
    # lenet architecture
        model = Sequential()
        shape = (height, width, depth)

        # first set of layers
        model.add(Conv2D(20, (5,5), padding='same', input_shape=shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # next set of layers
        model.add(Conv2D(50, (5,5), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # fully connect layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))

        # binary problem
        model.add(Dense(2))
        model.add(Activation('softmax'))

        return model


# init data and labels
data = []
labels = []


# preprocessing the images
for img_path in glob.glob('data/*/*.jpg'):
    # construct data
    im = Image.open(img_path)
    im = im.resize((28, 28))
    im_array = img_to_array(im)
    data.append(im_array)

    # extract labels
    label = img_path.split('/')[1]

    if label == "bird":
        l = 1
    else:
        l = 0
    
    labels.append(l)


dim_correct = [data[x].shape == (28, 28, 3) for x in range(0, len(data))]
data_sub = [i for (i, v) in zip(data, dim_correct) if v is True]
labels_sub = [i for (i, v) in zip(labels, dim_correct) if v is True]

# scaling to 0-1
data = np.array(data_sub, dtype='float') / 255.0
labels = np.array(labels_sub) 


#%%
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=122)

y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)


# some augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True, 
    fill_mode="nearest")


# modelling
epochs = 80
learning_rate = .001
batch_size = 32

model = LeNet.build_model(width=28, height=28, depth=3)
optimizer = Adam(lr=learning_rate, decay=learning_rate/epochs)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

datagen.fit(X_train)
H = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), steps_per_epoch=len(X_train)/batch_size, epochs=epochs, verbose=2)

model.save("bird_or_not.model")