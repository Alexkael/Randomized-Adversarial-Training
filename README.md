# Randomized Adversarial Training
Code for CVPR 2023 Paper [**'Randomized Adversarial Training via Taylor Expansion'**](https://arxiv.org/abs/2303.10653)

# Requisite
Python 3.6+  
Pytorch 1.8.1+cu111
Torchvision 0.9.0+cu111

# How to use
Run train_first_second.py  
We got the best performance between epoch 100-200

# Evaluation
PGD and CW evaluation with epsilon=0.031. Auto attack evaluation is under standard version with epsilon=8/255.

# Reference Code
[1] AT: https://github.com/locuslab/robust_overfitting  
[2] TRADES: https://github.com/yaodongyu/TRADES/  
[3] AutoAttack: https://github.com/fra31/auto-attack  
[4] MART: https://github.com/YisenWang/MART  
[5] AWP: https://github.com/csdongxian/AWP  
[6] AVMixup: https://github.com/hirokiadachi/Adversarial-vertex-mixup-pytorch  
[7] S2O: https://github.com/Alexkael/S2O
