#!/bin/bash

# vary windows and samples
w_array=(100 24 12 6)
s_array=(15 105 1005)
echo "vary weights"
for w in "${w_array[@]}"; do
	for s in "${s_array[@]}"; do
		python inspect_pong.py $w $s 
	done
done
