#!/bin/bash
echo Test
pytest test_algorithms.py

echo Run experiments
for GRID_SIDE in 5 6 7 8 9 10
do
	python run_train.py --learning_type=smallgan --grid_side_size=$GRID_SIDE
	python run_train.py --learning_type=gan --grid_side_size=$GRID_SIDE
done