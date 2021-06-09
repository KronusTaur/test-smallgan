#!/bin/bash
echo Small-GAN time
time python run_train.py --learning_type=smallgan --train_steps=1000 > temp.log

echo GAN time
time python run_train.py --learning_type=gan --train_steps=1000 > temp.log