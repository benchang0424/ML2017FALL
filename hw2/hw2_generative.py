import numpy as np
import sys
import csv
from numpy.linalg import inv

X_trainraw_path = sys.argv[1]
X_testraw_path = sys.argv[2]
X_train_path = sys.argv[3]
Y_train_path = sys.argv[4]
X_test_path = sys.argv[5]
X_predict_path =sys.argv[6] 

"load data"
X_train = np.genfromtxt(X_train_path, delimiter=',' , skip_header=1)
Y_train = np.genfromtxt(Y_train_path, delimiter=',' , skip_header=1)
Y_train = Y_train.reshape(32561,1)
X_test = np.genfromtxt(X_test_path, delimiter=',' , skip_header=1)


def normalization(A,avg,std):     
    return (A - avg) / std

def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z))

avg = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
t_avg = np.mean(X_test, axis=0)
t_std = np.std(X_test, axis=0)
X_train = normalization(X_train,avg,std)
X_test = normalization(X_test,avg,std)

X_class1 = np.zeros(shape=[7841, 106])		# > 50k
X_class0 = np.zeros(shape=[24720, 106])		# < 50k
idx1 = 0
idx0 = 0


for i in range(len(Y_train)):
	if(Y_train[i][0]==1):
		X_class1[idx1] = X_train[i]
		idx1 += 1
	else:
		X_class0[idx0] = X_train[i]
		idx0 += 1

class1_num = len(X_class1)	#num of >50k	#24720
class0_num = len(X_class0)					#7841

class1_mean = np.mean(X_class1, axis=0)
class0_mean = np.mean(X_class0, axis=0)

#class1_cov = np.cov(X_class1.transpose())
#class0_cov = np.cov(X_class0.transpose())
class1_cov = np.zeros(shape=(106,106))
class0_cov = np.zeros(shape=(106,106))

"===="

for r in range(class1_num):
    v = (X_class1[r]- class1_mean)
    class1_cov += v.reshape(106,1)*v.reshape(1,106)
for r in range(class0_num):
    v = (X_class0[r]- class0_mean)
    class0_cov += v.reshape(106,1)*v.reshape(1,106)

class1_cov /= class1_num
class0_cov /= class0_num

"===="

total_cov = class1_num/(class1_num+class0_num) * class1_cov + class0_num/(class1_num+class0_num) * class0_cov

w = ((class0_mean - class1_mean).dot(inv(total_cov)))
b = (- 1/2*(class0_mean.dot(inv(total_cov))).dot(class0_mean.transpose()) + 1/2* (class1_mean.dot(inv(total_cov))).dot(class1_mean.transpose()) + np.log(class0_num/class1_num))


answer_list = []
answer_list.append(['id' , 'label'])
for n in range(len(X_test)):
    if sigmoid(w.dot(X_test[n]) + b ) > 0.5:
        answer_list.append([str(int(n+1)),str(0)])
    else:
        answer_list.append([str(int(n+1)),str(1)])
        

file = open(X_predict_path,'w')
w = csv.writer(file)    
w.writerows(answer_list)
file.close()




