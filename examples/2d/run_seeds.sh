#!/bin/bash
for i in {0..2} 
do
  python cifar_resnet_torch.py --mode 1 --seed $i
  python cifar_resnet_torch.py --mode 2 --seed $i
done

