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

def normalization(A,avg,std):     
    return (A - avg) / std

def sigmoid(z):
	res = 1 / (1.0 + np.exp(-z))
	return np.clip(res, 0.00000000000001, 0.99999999999999)

def gradientDescent(X, Y, b, w, lr, numIterations, lamda):

    lr_b = 0
    lr_w = np.zeros((X_dim,1),dtype=float)
    b_grad = 0.0
    w_grad = np.zeros((X_dim,1),dtype=float)

    for i in range(0, numIterations):
        preds = sigmoid(b + X.dot(w))
        diff = Y - preds    # dim * 1 matrix
        
        b_grad = diff.sum() * (-1.0)
        w_grad = -1.0* X.transpose().dot(diff) + 2*lamda*w

        lr_b = lr_b + b_grad ** 2
        lr_w = lr_w + w_grad ** 2
        b = b - lr/np.sqrt(lr_b) * b_grad
        w = w - lr/np.sqrt(lr_w) * w_grad

        if(i%100==0):
        	loss = -np.sum(Y*np.log(preds)+(1-Y)*np.log(1-preds))
        	print("Iteration %d | Loss: %f" % (i, loss) + "\r",end="")
        
        
    return b,w

numIterations= 20000
lamda = 0.1
X_dim = 106+7
lr = 0.2
b = 1.0
w = np.ones((X_dim,1) ,dtype=float)


X_train = np.genfromtxt(X_train_path, delimiter=',' , skip_header=1)
X_train = np.c_[X_train, X_train[:,0]**2, X_train[:,1]**2, X_train[:,2]**2, X_train[:,3]**2, X_train[:,5]**2 ,X_train[:,0]**3,X_train[:,5]**3]

Y_train = np.genfromtxt(Y_train_path, delimiter=',' , skip_header=1)
Y_train = Y_train.reshape(32561,1)
#print(Y_train.shape)

X_test = np.genfromtxt(X_test_path, delimiter=',' , skip_header=1)
X_test = np.c_[X_test, X_test[:,0]**2, X_test[:,1]**2, X_test[:,2]**2,X_test[:,3]**2, X_test[:,5]**2, X_test[:,0]**3, X_test[:,5]**3]

#print(X_test.shape)


avg = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
t_avg = np.mean(X_test, axis=0)
t_std = np.std(X_test, axis=0)
X_train = normalization(X_train,avg,std)
X_test = normalization(X_test,avg,std)

b,w = gradientDescent(X_train, Y_train, b, w, lr, numIterations, lamda)
#print(b)
#print(w)
print()
answer_list = []
answer_list.append(['id' , 'label'])
for n in range(len(X_test)):
    if sigmoid(X_test[n].dot(w) + b ) >= 0.5:
        answer_list.append([str(int(n+1)),str(1)])
    else:
        answer_list.append([str(int(n+1)),str(0)])
        

file = open(X_predict_path,'w')
w = csv.writer(file)    
w.writerows(answer_list)
file.close()
