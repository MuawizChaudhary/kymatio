#!/bin/bash
for i in {0..2} 
do
  for j in {0..2}
  do
    python cifar_small_sample.py --mode 'scattering' --seed $i --dataset_seed $j
    python cifar_small_sample.py --mode 'standard' --seed $i --dataset_seed $j
  done
done
