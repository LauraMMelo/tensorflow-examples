#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 10:36:35 2019

@author: computervision
"""

import numpy as np
from keras.utils import np_utils, Sequence
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from sklearn.metrics import classification_report
from PIL import Image
import pandas as pd
import cv2 
from cv2 import resize
from matplotlib.backends.backend_pdf import PdfPages
import glob
from keras import applications
import os

#Hiperparâmetros
batch_size = 32
epochs = 1
num_classes = 53

##############PRÉ-PROCESSAMENTO###################

class My_generator(Sequence):
    
    def __init__(self, image_filenames, labels, batch_size):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size
    
    def __len__(self):
        return np.ceil(len(self.image_filenames)/float(self.batch_size)).astype(np.int64)
    
    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        return np.array([
                resize(mpimg.imread(file_name), (299,299))/255.
                    for file_name in batch_x]), np.array(batch_y)


#Transformando imagense classes em numpy arrays

images = []
cars = []
classes1 = []
path = '/home/computervision/Vídeos/nosso_db/cropped/'
i = 0
for name in os.listdir(path):
    for item in os.listdir(path + name):


#        orig_img = mpimg.imread(path + name + "/" + item)
#        rsz_img = resize(orig_img, (299,299))
        images.append(path + name + "/" + item)
        cars.append(name)
        classes1.append(i)
    
    i = i+1
        
############# Transformando em numpy arrays ############
images = np.array(images)
classes1 = np.array(classes1, dtype=int)
#
classes = np_utils.to_categorical(classes1)


train_X,test_X,train_label,test_label = train_test_split(images, classes, test_size=0.2, random_state=13)

train_X.shape,test_X.shape,train_label.shape,test_label.shape

train_X, valid_X, train_label, valid_label = train_test_split(train_X, train_label, test_size=0.2, random_state=13)

train_X.shape, valid_X.shape, train_label.shape, valid_label.shape


my_training_batch_generator = My_generator(train_X, train_label, batch_size)
my_validation_batch_generator = My_generator(valid_X, valid_label, batch_size)

num_training_samples = train_X.shape[0]
num_validation_samples = valid_X.shape[0]


#Instanciando modelo pré-treinado
def inception(use_imagenet=True):
    # load pre-trained model graph, don't add final layer
    model = keras.applications.InceptionV3(include_top=False, input_shape=(299, 299, 3),
                                          weights='imagenet' if use_imagenet else None)
    # add global pooling just like in InceptionV3
    new_output = keras.layers.GlobalAveragePooling2D()(model.output)
    # add new dense layer for our labels
    new_output = keras.layers.Dense(num_classes, activation='softmax')(new_output)
    model = keras.engine.training.Model(model.inputs, new_output)
    return model


#Sumário
model = inception()
print(len(model.layers))

#trainable layers
# set all layers trainable by default
for layer in model.layers:
    layer.trainable = True
    if isinstance(layer, keras.layers.BatchNormalization):
        # we do aggressive exponential smoothing of batch norm
        # parameters to faster adjust to our new dataset
        layer.momentum = 0.9
    
# fix deep layers (fine-tuning only last 50)
for layer in model.layers[:-50]:
    # fix all but batch norm layers, because we neeed to update moving averages for a new dataset!
    if not isinstance(layer, keras.layers.BatchNormalization):
        layer.trainable = False

model.summary()

# Treinamento
adamax = keras.optimizers.Adamax(lr=0.01, 
                             beta_1=0.9, 
                             beta_2=0.999, 
                             epsilon=None, 
                             decay=0.0)

model.compile(loss=keras.losses.categorical_crossentropy,
                        optimizer= adamax,
                        metrics=['accuracy'])

#treinando o modelo com o novo dataset
car_color_train_dropout = model.fit_generator(generator=my_training_batch_generator,
                                              steps_per_epoch= num_training_samples // batch_size // 8,
                                              epochs=epochs,
                                              verbose=1,
                                              validation_data=my_validation_batch_generator,
                                              validation_steps= num_validation_samples // batch_size // 4)


#Salvando
model.save("car_brand_model_Inceptionv3.h5py")



##################### RESULTADOS ##############3
test_img = []

for item in test_X:
	orig_img = mpimg.imread(item)
	rsz_img = resize(orig_img, (299,299))
	test_img.append(rsz_img)

test_img = np.array(test_img)
test_img = test_img/255


test_eval = model.evaluate(test_img, test_label, verbose=1)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

predicted_classes = model.predict(test_img)

predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
labels = np.argmax(test_label, axis = 1, out = None)

predicted_classes.shape, labels.shape    
    
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(labels, predicted_classes, target_names=target_names))

