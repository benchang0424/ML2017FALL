import numpy as np
import matplotlib.pyplot as plt
import csv
import sys

input_data = np.genfromtxt('train.csv', delimiter=',')
d = input_data[1:,3:]
d[np.isnan(d)] = 0
data = np.empty(shape=[18, 24*20*12])

for i in range(1, 20*12+1):
    data[:, 24*(i-1) : 24*i] = d[18*(i-1):18*i , :]

# data[9] = PM2.5

"=== parameters ==="

select_hour = 9
select_item = 3         #ex: O3, pm10, pm2_5, pm2_5 ^2
i_dim = 24*20*12
numIterations= 50000
lamda = 0.01
X_dim = select_item * select_hour
lr = 0.7
b = 0.0
w = np.ones((X_dim,1) ,dtype=float)

" =============================   ============================== "

def gradientDescent(X, Y, b, w, lr, numIterations, lamda):

    lr_b = 0
    lr_w = np.zeros((X_dim,1),dtype=float)
    b_grad = 0.0
    w_grad = np.zeros((X_dim,1),dtype=float)
    for i in range(0, numIterations):
        preds = b + X.dot(w)
        diff = Y - preds    # dim * 1 matrix
        loss = np.sqrt((diff**2).sum()/X.shape[0])
        
        b_grad = -2.0* diff.sum() *1.0
        w_grad = -2.0* X.transpose().dot(diff) + 2*lamda*w

        lr_b = lr_b + b_grad ** 2
        lr_w = lr_w + w_grad ** 2
        b = b - lr/np.sqrt(lr_b) * b_grad
        w = w - lr/np.sqrt(lr_w) * w_grad

        
        print("Iteration %d | Loss: %f" % (i, loss) + "\r",end="")
        
    return b,w

" =============================   ============================== "

def feature_scaling(A,avg,std):		
	return (A - avg) / std

def normalize(X,x_max):
    return X / x_max

"===== build training data ====="

X = np.empty(shape=[0, X_dim])
Y = np.empty(shape=[0, 1])

for i in range(0,i_dim):
    if (i%480 >= select_hour):
        select_x0 = data[ 7 , i-select_hour :i ]

        select_x1 = data[ 9 , i-select_hour :i ]
        x_square = select_x1 ** 2
        select_x = np.vstack([select_x0,select_x1, x_square])
        y = data[9][i].reshape(1,1)
        #print(original_x)
        x_row = select_x.reshape(1,X_dim)
        X = np.append(X, x_row, axis=0)
        Y = np.append(Y, y, axis=0)

avg = np.mean(X, axis=0)
std = np.std(X, axis=0)
x_max = X.max(axis=0)
#train = feature_scaling(X,avg,std)
train = normalize(X,x_max)
"============ train =============="

b,w = gradientDescent(train,Y,b,w,lr,numIterations,lamda)


#np.savetxt('w.csv',w)

"=== Test data read in ==="

test_input_data = np.genfromtxt(sys.argv[1], delimiter=',')
test_data = test_input_data[:,2:]
test_data[np.isnan(test_data)] = 0

rows = test_data.shape[0]  #240*18


_list = []
_list.append(['id' , 'value'])

tX = np.empty(shape=[0, X_dim])
for i in range(0,rows,18):
    x0 = test_data[7+i, 9-select_hour:]
    x1 = test_data[9+i, 9-select_hour:]            #.reshape(15,1)
    x_square = x1 ** 2
    x_row = np.vstack([x0, x1, x_square]).reshape(1,X_dim)
    #x = np.vstack([x0, x1, x_square]).reshape(X_dim,1)
    tX = np.append(tX, x_row, axis=0)

    

#test_X = feature_scaling(tX,avg,std)
test_X = normalize(tX,x_max)
pred_Y = b + test_X.dot(w)

for i in range(len(pred_Y)):
	id_ = "id_" + str(int(i))
	outdata = [id_, str(float(pred_Y[i]))]
	_list.append(outdata)



"=== output result ==="

file = open(sys.argv[2],'w')
w = csv.writer(file)    
w.writerows(_list)
file.close()


