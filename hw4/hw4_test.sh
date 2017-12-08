#!/bin/bash
wget -O model.bin https://www.dropbox.com/s/4i24j01o99k4vwp/model.bin?dl=1
wget -O tokenizer.pkl https://www.dropbox.com/s/m8gas7vr2s7vqm2/tokenizer.pkl?dl=1
wget -O best_model.h5 https://www.dropbox.com/s/knrua1dzxj52gmd/best_model.h5?dl=1
python3 test.py $1 $2