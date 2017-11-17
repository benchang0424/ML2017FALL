#!/bin/bash
python3 hw3_train.py $1 model_train.h5
cp model_train.h5 model_1.h5
cp model_train.h5 model_2.h5
cp model_train.h5 model_3.h5
cp model_train.h5 model_4.h5