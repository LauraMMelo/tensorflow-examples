#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 13:57:30 2018

@author: computervision
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.optimizers import SGD, Adam
from keras.losses import categorical_crossentropy
from keras.models import Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, concatenate
from keras.utils import plot_model
from keras.utils import np_utils, Sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from cv2 import resize
import os


num_classes = 8
batch_size = 115
epochs = 20

################## DATA HANDLER ################

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
                resize(mpimg.imread(file_name), (227,227))/255.
                    for file_name in batch_x]), np.array(batch_y)



################ LOADING IMAGES ################
images = []
colors = []
classes = []
path = '/home/bimo/Documentos/Saulo/color/'

for name in os.listdir(path):
    for item in os.listdir(path + name):
#        orig_img = mpimg.imread(path + name + "/" + item)
#        rsz_img = resize(orig_img, (227,227))
        images.append(path + name + "/" + item)
        colors.append(name)

############### CONVERTING TO INT ################
#black, clue, cyan, gray, green, red, white, yellow        
for element in range(len(colors)):
    if colors[element] == 'black':
        classes.append(0)
    if colors[element] == 'blue':
        classes.append(1)
    if colors[element] == 'cyan':
        classes.append(2)
    if colors[element] == 'gray':
        classes.append(3)
    if colors[element] == 'green':
        classes.append(4)
    if colors[element] == 'red':
        classes.append(5)
    if colors[element] == 'white':
        classes.append(6)
    if colors[element] == 'yellow':
        classes.append(7)

############### CONVERT TO NP ARRAYS ###############

images = np.array(images)
classes = np.array(classes, dtype=int)
#
#train_X = images[0:12480,:,:,:]
#train_Y = classes[0:12480]
#test_X = images[12481:15601,:,:,:]
#test_Y = classes[12481:15601]
#
#train_X = train_X.astype('float32')
#test_X = test_X.astype('float32')
#train_X = train_X/255.
#test_X = test_X/255.  
#

#transformando classes em vetor binário
#train_Y_one_hot = np_utils.to_categorical(train_Y)
#test_Y_one_hot = np_utils.to_categorical(test_Y)
#
        
classes = np_utils.to_categorical(classes)


train_X,test_X,train_label,test_label = train_test_split(images, classes, test_size=0.2, random_state=13)

train_X.shape,test_X.shape,train_label.shape,test_label.shape

train_X, valid_X, train_label, valid_label = train_test_split(train_X, train_label, test_size=0.2, random_state=13)

train_X.shape, valid_X.shape, train_label.shape, valid_label.shape


my_training_batch_generator = My_generator(train_X, train_label, batch_size)
my_validation_batch_generator = My_generator(valid_X, valid_label, batch_size)

num_training_samples = train_X.shape[0]
num_validation_samples = valid_X.shape[0]

    
################## MODEL #######################

#num_classes = 8
#batch_size = 115
#epochs = 30

main_input = Input(shape=(227,227,3), name='main_input')

conv1_1 = Conv2D(48,strides=4, kernel_size=(11,11), activation='relu')(main_input)
maxpool1_1 = MaxPooling2D(pool_size=(3,3), strides=2)(conv1_1)
conv1_2 = Conv2D(48,strides=4, kernel_size=(11,11), activation='relu')(main_input)
maxpool1_2 = MaxPooling2D(pool_size=(3,3), strides=2)(conv1_2)

conv2_1 = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(maxpool1_1)
maxpool2_1 = MaxPooling2D(pool_size=(3,3), strides=2)(conv2_1)
conv2_2 = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(maxpool1_1)
maxpool2_2 = MaxPooling2D(pool_size=(3,3), strides=2)(conv2_2)

conv2_3 = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(maxpool1_2)
maxpool2_3 = MaxPooling2D(pool_size=(3,3), strides=2)(conv2_3)
conv2_4 = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(maxpool1_2)
maxpool2_4 = MaxPooling2D(pool_size=(3,3), strides=2)(conv2_4)

conv3_1 = Conv2D(96, kernel_size=(3,3), activation='relu', padding='same')(maxpool2_1)
conv3_2 = Conv2D(96, kernel_size=(3,3), activation='relu', padding='same')(maxpool2_2)
conv3_3 = Conv2D(96, kernel_size=(3,3), activation='relu', padding='same')(maxpool2_3)
conv3_4 = Conv2D(96, kernel_size=(3,3), activation='relu', padding='same')(maxpool2_4)

conc3_1 = concatenate([conv3_1, conv3_2])
conc3_2 = concatenate([conv3_3, conv3_4])

conv4_1 = Conv2D(96, kernel_size=(3,3), activation='relu', padding='same')(conc3_1)
conv4_2 = Conv2D(96, kernel_size=(3,3), activation='relu', padding='same')(conc3_1)
conv4_3 = Conv2D(96, kernel_size=(3,3), activation='relu', padding='same')(conc3_2)
conv4_4 = Conv2D(96, kernel_size=(3,3), activation='relu', padding='same')(conc3_2)

conv5_1 = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(conv4_1)
maxpool5_1 = MaxPooling2D(pool_size=(3,3), strides=2)(conv5_1)
flt5_1 = Flatten()(maxpool5_1) 
conv5_2 = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(conv4_2)
maxpool5_2 = MaxPooling2D(pool_size=(3,3), strides=2)(conv5_2)
flt5_2 = Flatten()(maxpool5_2) 
conv5_3 = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(conv4_3)
maxpool5_3 = MaxPooling2D(pool_size=(3,3), strides=2)(conv5_3)
flt5_3 = Flatten()(maxpool5_3) 
conv5_4 = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(conv4_4)
maxpool5_4 = MaxPooling2D(pool_size=(3,3), strides=2)(conv5_4)
flt5_4 = Flatten()(maxpool5_4) 

conc5_1 = concatenate([flt5_1, flt5_2, flt5_3, flt5_4])

fc1 = Dense(4096, activation='relu')(conc5_1)
drop1 = Dropout(0.3)(fc1)
fc2 = Dense(4096, activation='relu')(drop1)
drop2 = Dropout(0.3)(fc2)
fc3 = Dense(4096, activation='relu')(drop2)
drop3 = Dropout(0.3)(fc3)
output = Dense(num_classes, activation='softmax')(drop3)

model = Model(inputs=[main_input], outputs=[output])

#plot_model(model, to_file = 'Rachmadi_model.png', show_shapes = True)

######### Summary #####################

model.summary()


#################### TRAINING ################################33

sgd = SGD(lr=0.01,
		  momentum=0.9,
		  decay=0.0005)

model.compile(loss=categorical_crossentropy,
              optimizer= sgd,
              metrics=['accuracy'])

#car_color_train_dropout = model.fit(train_X, 
#                                train_label, 
#                                batch_size=batch_size, 
#                                epochs=epochs, 
#                                verbose=1, 
#                                validation_data=(valid_X, valid_label))

car_color_train_dropout = model.fit_generator(generator=my_training_batch_generator,
                                              steps_per_epoch=(num_training_samples // batch_size),
                                              epochs=epochs,
                                              verbose=1,
                                              validation_data=my_validation_batch_generator,
                                              validation_steps=(num_validation_samples // batch_size),
                                              use_multiprocessing=True,
                                              workers=16,
                                              max_queue_size=32)

model.save("rachmadi_model.h5py")

#car_color_model = keras.models.load_model("car_color_model_dropout.h5py")




#print('Test loss:', test_eval[0])
#print('Test accuracy:', test_eval[1])
#
#accuracy = car_color_train_dropout.history['acc']
#val_accuracy = car_color_train_dropout.history['val_acc']
#loss = car_color_train_dropout.history['loss']
#val_loss = car_color_train_dropout.history['val_loss']
#epochs = range(len(accuracy))
#plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
#plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
#plt.title('Training and validation accuracy')
#plt.legend()
#plt.figure()
#plt.plot(epochs, loss, 'bo', label='Training loss')
#plt.plot(epochs, val_loss, 'b', label='Validation loss')
#plt.title('Training and validation loss')
#plt.legend()
#plt.show()

test_img = []

for item in test_X:
	orig_img = mpimg.imread(item)
	rsz_img = resize(orig_img, (227,227))
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

#alguns resultados corretos
#correct = np.where(predicted_classes==test_Y)[0]
#print("Found %d correct labels" % len(correct))
#for i, correct in enumerate(correct[:9]):
#    plt.subplot(3,3,i+1)
#    plt.imshow(test_X[correct].reshape(28,28,3), cmap='gray', interpolation='none')
#    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_Y[correct]))
#    plt.tight_layout()
#
##alguns resultados incorretos    
#incorrect = np.where(predicted_classes!=test_Y)[0]
#print("Found %d incorrect labels" % len(incorrect))
#for i, incorrect in enumerate(incorrect[:9]):
#    plt.subplot(3,3,i+1)
#    plt.imshow(test_X[incorrect].reshape(28,28,3), cmap='gray', interpolation='none')
#    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], test_Y[incorrect]))
#    plt.tight_layout()

#Report    
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(labels, predicted_classes, target_names=target_names))
