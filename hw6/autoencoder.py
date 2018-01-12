import matplotlib
#matplotlib.use('Agg')
import numpy as np
import os
import sys
import csv
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from keras.layers import Dense, Input
from keras.models import Model, load_model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
IMAGE_PATH = sys.argv[1]
TEST_PATH = sys.argv[2]
PREDICT_PATH = sys.argv[3]

def read_test():
	test = []
	f = open(TEST_PATH,'r',encoding='utf8')
	next(f)
	for row in f :
		ID, img1, img2 = row[:-1].split(',')
		test.append([int(img1), int(img2)])
	f.close()
	return np.array(test)

test = read_test()

train=np.load(IMAGE_PATH) # 140000*784

print(test.shape)
print(train.shape)

mean = train.mean(axis=0)
train = train-mean
train = train.astype('float32') / 255

"""
encoding_dim = 64

input_img = Input(shape=(784,))
encoded = Dense(256, activation='relu')(input_img)
encoded = Dense(128, activation='relu')(encoded)
enocoded_output = Dense(encoding_dim)(encoded)
decoded = Dense(128, activation='relu')(enocoded_output)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(784, activation='tanh')(decoded)

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, enocoded_output)

opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0,clipvalue=0.5)
autoencoder.compile(optimizer=opt, loss='mse')

autoencoder.fit(train, train,
                epochs=60,
                batch_size=512,
                shuffle=True)

autoencoder.save('ae.h5')
encoder.save('encoder.h5')
"""
encoder = load_model('encoder.h5')
encoded_img = encoder.predict(train)
#encoded_img = PCA(n_components=32, whiten=True, random_state=424).fit_transform(encoded_img)
k_means = KMeans(n_clusters=2, random_state=666).fit(encoded_img)
predict = k_means.labels_

f = open(PREDICT_PATH,'w')
writer = csv.writer(f,delimiter=',',lineterminator='\n')
writer.writerow(["ID","ANS"])
print (test.shape[0])
for i in range(1980000):
	a=test[i][0]
	b=test[i][1]
	#print("a = %d , b = %d" %(predict[a],predict[b]))
	if(predict[a] == predict[b]):
		writer.writerow([i,str(1)])
	else:
		writer.writerow([i,str(0)])

