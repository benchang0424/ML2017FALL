import numpy as np
import skimage
from skimage import io
import os
import sys

images_path = sys.argv[1]
target_path = sys.argv[2]

def plot(data):
	data -= np.min(data)
	data /= np.max(data)
	data = (data*255).astype(np.uint8).reshape(600,600,3)
	io.imsave('reconstruction.jpg', data)

def pca(U,img,mean,k):
	img = img - mean
	out = U[:,:k].T.dot(img)
	return out, mean

def reconstruct(U,weight,mean,k):
	img = U[:,:k].dot(weight)
	img = img + mean
	return img

imgs = []
for i in range(415):
	filename = images_path +'/'+ str(i) + '.jpg'
	im = io.imread(filename)
	im = np.array(im)
	im = im.reshape(-1,)
	imgs.append(im)
imgs = np.array(imgs)
#imgs = imgs.T
print(imgs.shape)
X_mean = imgs.mean(axis=0)

X = imgs-X_mean
X = X.T			#1080000*415
print(X.shape)
U, s, Vt = np.linalg.svd(X, full_matrices=False)

print(U)

target_img = io.imread(os.path.join(sys.argv[1],sys.argv[2])).flatten()
weight, mean = pca(U,target_img,X_mean,4)
re_img = reconstruct(U,weight,X_mean,4)
plot(re_img)



