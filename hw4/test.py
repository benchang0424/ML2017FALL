import numpy as np
import pandas as pd
import sys
import csv
import gensim
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
import keras
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Conv1D, Flatten, Dropout, Dense, Activation ,GRU
from keras.layers import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD, Adam, Adadelta
from keras.models import Sequential
from keras.models import load_model
from keras.utils import np_utils
import pickle as pk

X,Y,sentences = [],[],[]
x_train,y_train = [],[]
x_val,y_val = [],[]
x_test = []

# parameters
embedding_vecor_length = 100
EMBEDDING_DIM = 200
BATCH_SIZE = 512
"""
def read_training_labeldata():
	data = pd.read_csv('training_label.txt', delimiter = '\+\+\+\$\+\+\+',header = None)
	Y = data.loc[:,0].as_matrix().reshape(200000,1)
	#X = data.loc[:,1].astype(str).as_matrix()
	X = data.loc[:,1].astype(str).values.tolist()
	sentences = data.loc[:,1].astype(str).values.tolist()
	print("Fininshing reading training data")
	return X,Y,sentences

def read_training_nolabeldata():
	f = open('training_nolabel.txt','r',encoding='utf8')
	for row in f :
		tmp_c = row.split()
		sentences.append(tmp_c)
	f.close()

def read_testing_data():
	data = pd.read_csv('testing_data.txt',sep=',',index_col=0,header=0, usecols=["id", "text"])
	data = data.astype(str).values.tolist()
	print("Fininshing reading testing data")
	return data
"""
"""
f = open('training_label.txt','r',encoding='utf8')
for row in f :
	tmp = row.split(' +++$+++ ')
	tmp[1] = tmp[1][:-1]
	x_train.append(tmp[1])
	y_train.append(int(tmp[0]))
	sentences.append(tmp[1].split(' '))
f.close()

f = open('training_nolabel.txt','r',encoding='utf8')
for row in f :
	tmp = row.split()
	sentences.append(tmp)
f.close()
"""
f = open(sys.argv[1],'r',encoding='utf8')
next(f)
for row in f :
	idx , content = row.split(',',1)
	content = content[:-1]	
	x_test.append(content)
f.close()

print("========== Fininshing reading data ==========")

# train model
#model = Word2Vec(sentences,size = EMBEDDING_DIM,min_count=10)
# summarize the loaded model
#print(model)
# summarize vocabulary
#words = list(model.wv.vocab)
#print(len(words))
#print(words)
#t = Tokenizer()
#t.fit_on_texts(x_train+x_test)
#print(t.word_index)
#input()


model = Word2Vec.load('model.bin')

#t = Tokenizer()
t = pk.load(open('tokenizer.pkl','rb'))


#with open('tokenizer.pkl', 'wb') as handle:
#    pickle.dump(t, handle)

vocab_size = len(t.word_index) + 1
print (vocab_size)
print (len(x_test))

x_test = t.texts_to_sequences(x_test)

max_length = 20

x_test = pad_sequences(x_test,maxlen=max_length,padding='post')

x_test = np.array(x_test)


"""

embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
for word, i in t.word_index.items():
	if word in model :
		embedding_vector = model[word]
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector
#	else : 
#		embedding_matrix[i] = embedding_matrix[0]


# save model
#model.save('model.bin')

embedding_layer = Embedding(vocab_size,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=max_length,
                            trainable=False)


rnn = Sequential()
#rnn.add(Embedding(vocab_size, embedding_vecor_length, input_length=max_length))
#rnn.add(Conv1D(64, 3, border_mode='same'))
#rnn.add(Conv1D(32, 3, border_mode='same'))
#rnn.add(Conv1D(16, 3, border_mode='same'))
rnn.add(embedding_layer)
rnn.add(GRU(512, recurrent_dropout = 0.3, dropout=0.3, return_sequences=True, activation='relu'))
rnn.add(BatchNormalization())
rnn.add(GRU(256, recurrent_dropout = 0.3, dropout=0.3, return_sequences=True, activation='relu'))
rnn.add(BatchNormalization())
rnn.add(GRU(128, recurrent_dropout = 0.3, dropout=0.3, activation='relu'))
rnn.add(BatchNormalization())
rnn.add(Dropout(0.4))
rnn.add(Dense(256,activation='relu'))
rnn.add(BatchNormalization())
rnn.add(Dropout(0.4))
rnn.add(Dense(128,activation='relu'))
rnn.add(BatchNormalization())
rnn.add(Dropout(0.4))
rnn.add(Dense(1,activation='sigmoid'))
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0,clipvalue=0.5) #initial:lr=0.001

rnn.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
rnn.summary()

checkpoint = ModelCheckpoint('best_model.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=0, mode='auto')

train_history = rnn.fit(x_train, y_train, validation_data=(x_val, y_val)
				,callbacks = [early_stopping,checkpoint]
				,epochs=50, batch_size=BATCH_SIZE, verbose=1)

with open('training_procedure_rnn.pkl', 'wb') as handle:
    pickle.dump(train_history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""


predict_model = load_model('best_model.h5')
predict_model.summary()
y_test = predict_model.predict(x_test,batch_size=BATCH_SIZE, verbose=1)

f = open(sys.argv[2],'w')
writer = csv.writer(f,delimiter=',',lineterminator='\n')
writer.writerow(["id","label"])
for i in range(len(y_test)):
    #line = y_test[i].tolist()
    if(y_test[i] >= 0.5):
    	writer.writerow([i,str(1)])
    else : 
    	writer.writerow([i,str(0)])

