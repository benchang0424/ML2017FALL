import numpy as np
import pandas as pd
import sys
import csv
import keras
import keras.backend as K
from keras.utils import np_utils
from keras.layers import Input, Dense, Embedding, Dropout, Flatten
from keras.models import Model
from keras.layers.merge import Dot, Add, Concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model, load_model
from keras.optimizers import SGD, Adam, Adadelta


TEST_PATH = sys.argv[1]
OUTPUT_PATH = sys.argv[2]
MOVIE_PATH = sys.argv[3]
USER_PATH = sys.argv[4]

#EMBEDDING_DIM = 256
BATCH_SIZE = 2048

train_data, test_data = [],[]

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def read_user():
	genders, ages, occus = (np.zeros((6041,)) for i in range(3)) 
	f = open(USER_PATH,'r',encoding='utf8')
	next(f)
	for row in f :
		userID, gender, age, occu, zipcode = row[:-1].split('::')
		genders[int(userID)] = 0 if gender=='F' else 1
		ages[int(userID)] = int(age)
		occus[int(userID)] = int(occu)
	f.close()
	occus = np_utils.to_categorical(occus,)
	return genders,ages,occus

def read_movie():
	all_genre = ['Action','Adventure','Animation','Children\'s','Comedy','Crime'
				,'Documentary','Drama','Fantasy','Film-Noir','Horror','Musical'
				,'Mystery','Romance','Sci-Fi','Thriller','War','Western']
	movies = np.zeros((3953,18))

	f = open(MOVIE_PATH,'r',encoding='latin-1')
	next(f)
	for row in f :
		movieID, title, genre = row[:-1].split('::')
		for g in genre.split('|'):
			movies[int(movieID)][all_genre.index(g)] = 1
	f.close()
	return movies

def read_test():
	test_data = []
	f = open(TEST_PATH,'r',encoding='utf8')
	next(f)
	for row in f :
		dataID, userID, movieID = row[:-1].split(',')
		test_data.append([int(dataID), int(userID), int(movieID)])
	f.close()
	return test_data

genders,ages,occus = read_user()
moviesgenre = read_movie()
test_data = read_test()
test_data = np.array(test_data) 

test_userID = test_data[:,1]
test_movieID = test_data[:,2]
test_genders = genders[test_userID]
test_ages = ages[test_userID]
test_occus = occus[test_userID]
test_moviesgenre = moviesgenre[test_movieID]

model1 = load_model('models/MF_64.h5', custom_objects={'rmse': rmse})
model2 = load_model('models/MF_128.h5', custom_objects={'rmse': rmse})
model3 = load_model('models/MF_256.h5', custom_objects={'rmse': rmse})
model4 = load_model('models/MF_512.h5', custom_objects={'rmse': rmse})
model5 = load_model('models/modelmu_b.h5', custom_objects={'rmse': rmse})


result1 = model1.predict([test_userID,test_movieID],batch_size=BATCH_SIZE, verbose=1)
print()
result2 = model2.predict([test_userID,test_movieID],batch_size=BATCH_SIZE, verbose=1)
print()
result3 = model3.predict([test_userID,test_movieID],batch_size=BATCH_SIZE, verbose=1)
print()
result4 = model4.predict([test_userID,test_movieID],batch_size=BATCH_SIZE, verbose=1)
print()
result5 = model5.predict([test_userID,test_genders,test_ages,test_occus,test_movieID,test_moviesgenre],batch_size=BATCH_SIZE, verbose=1)
print()

result = (result1+result2+result3+result4+result5) / 5
rating = np.clip(result, 1, 5)
print(rating)

f = open(OUTPUT_PATH,'w')
writer = csv.writer(f,delimiter=',',lineterminator='\n')
writer.writerow(["TestDataID","Rating"])
for i in range(len(rating)):
    writer.writerow([i+1,str(rating[i][0])])
