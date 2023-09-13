#!/bin/bash
for i in {0..2} 
do
  python cifar_torch.py --mode 1 --classifier linear --seed $i
  python cifar_torch.py --mode 1 --classifier mlp --seed $i
  python cifar_torch.py --mode 1 --classifier cnn --seed $i
  python cifar_torch.py --mode 2 --classifier linear --seed $i
  python cifar_torch.py --mode 2 --classifier mlp --seed $i
  python cifar_torch.py --mode 2 --classifier cnn --seed $i
done

