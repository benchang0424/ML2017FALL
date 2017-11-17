import numpy as np
import pandas as pd
import sys
import csv
from keras.utils import np_utils
from keras.models import load_model


raw_test  = np.genfromtxt(sys.argv[1], delimiter=',' , dtype=str, skip_header=1)
x_test = []
for i in range(len(raw_test)):
	x_test.append(np.array(raw_test[i,1].split(' ')).reshape(48,48,1))
x_test = np.array(x_test, dtype=float) / 255

model1 = load_model('model_1.h5')
model2 = load_model('model_2.h5')
model3 = load_model('model_3.h5')
model4 = load_model('model_4.h5')

y1 = model1.predict(x_test, batch_size=128, verbose=1)
y2 = model2.predict(x_test, batch_size=128, verbose=1)
y3 = model3.predict(x_test, batch_size=128, verbose=1)
y4 = model4.predict(x_test, batch_size=128, verbose=1)
#y10 = model10.predict(x_test, batch_size=128, verbose=1)
#y11 = model11.predict(x_test, batch_size=128, verbose=1)

y = y1+y2+y3+y4
#y = y1 + y2 + y3 + y7 + y8 + y9 + y10 + y11
#y = y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8 + y9

f = open(sys.argv[2],'w+')
writer = csv.writer(f,delimiter=',',lineterminator='\n')
writer.writerow(["id","label"])

for i in range(len(y)):
    line = y[i].tolist()
    writer.writerow([i,line.index(max(line))])
