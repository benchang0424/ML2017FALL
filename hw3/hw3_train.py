import numpy as np
import pandas as pd
import keras
import csv
import sys
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam, Adadelta
import matplotlib.pyplot as plt
import pickle
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf  
import os

""" 

x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_val = np.load('x_val.npy')
y_val = np.load('y_val.npy')
x_test = np.load('x_test.npy')
"""

"============== set parameters ===================="
train_data = sys.argv[1]
model_best_path = sys.argv[2]
BATCH_SIZE = 512
"============== set data =========================="

raw_train = np.genfromtxt(train_data, delimiter=',' , dtype=str, skip_header=1)   #(28709, 2)
#raw_test  = np.genfromtxt('test.csv', delimiter=',' , dtype=str, skip_header=1)    #(7178, 2)

"=========== construct training/validation/testing data ============="

x_train, y_train, x_val, y_val = [],[],[],[]

for i in range(len(raw_train)):
	tmp = np.array(raw_train[i,1].split(' ')).reshape(48,48,1)
	if (i%100<=4):
		x_val.append(tmp)
		y_val.append(raw_train[i][0])
	else:
		x_train.append(tmp)
		x_train.append(np.flip(tmp,axis=1))
		y_train.append(raw_train[i][0])
		y_train.append(raw_train[i][0])

x_train = np.array(x_train, dtype=float) / 255
y_train = np.array(y_train, dtype=int)
x_val = np.array(x_val, dtype=float) / 255
y_val = np.array(y_val, dtype=int)

y_train = np_utils.to_categorical(y_train,7)
y_val = np_utils.to_categorical(y_val,7)


"============== data generator ===================="

datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False)
datagen.fit(x_train)
#datagen.fit(x_val)

"============== build training model =============="

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1),padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
model.add(Dropout(0.3))
model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(256, (3, 3), activation='relu',padding='same'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
model.add(Conv2D(256, (3, 3), activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.3))


model.add(Flatten())
model.add(Dropout(0.4))

for i in range(1):
	model.add(Dense(units=512, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))

for i in range(1):
	model.add(Dense(units=1024, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.45))

model.add(Dense(units=7, activation='softmax'))

#opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) #initial:lr=0.001
#opt = Adadelta(lr=0.5, rho=0.95, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()


"================================================="

checkpoint = ModelCheckpoint(model_best_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto')

model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE,shuffle=True)
			,validation_data=(x_val, y_val)
			,steps_per_epoch=len(x_train) // BATCH_SIZE
			,callbacks=[checkpoint]
			,epochs=300)


"================================================="
#model.save(model_path)

"================================================="
#y_test = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0)

#np.save(test_y_path, y_test)

#output = np.load(test_y_path)


"""
# write ouput file
f = open(out_path,'w')
writer = csv.writer(f,delimiter=',',lineterminator='\n')
writer.writerow(["id","label"])

for i in range(len(output)):
    line = output[i].tolist()
    writer.writerow([i,line.index(max(line))])
"""