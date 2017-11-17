#!/bin/bash
wget -O model_1.h5 https://www.dropbox.com/s/ln1kq9332ctlu6r/m03_2.h5?dl=1
wget -O model_2.h5 https://www.dropbox.com/s/20tbazi43pxkajx/m03_7best.h5?dl=1
wget -O model_3.h5 https://www.dropbox.com/s/i69ugznvb64a9yx/m03_8best.h5?dl=1
wget -O model_4.h5 https://www.dropbox.com/s/a1acnjluqkne1dc/m03_9best.h5?dl=1
python3 y_predict.py $1 $2