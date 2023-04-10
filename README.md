# <p align="center">Randomized Adversarial Training</p>
[![Arxiv](https://img.shields.io/badge/Arxiv-red.svg)](https://arxiv.org/abs/2303.10653)
[![Pub](https://img.shields.io/badge/Pub-CVPR'23-blue.svg)](https://arxiv.org/abs/2303.10653)
![License](https://img.shields.io/badge/license-MIT-yellow)

# Requisite
![正常大小的圆角矩形](https://img.shields.io/badge/Python-3.6+-green.svg)  
![正常大小的圆角矩形](https://img.shields.io/badge/Pytorch-1.8.1+cu111-green.svg)  
![正常大小的圆角矩形](https://img.shields.io/badge/Torchvision-0.9.0+cu111-green.svg)  

# How to use
Run AWP_first_second/train_first_second.py for randomized adversarial training based on AWP-TRADES.   
Run TRADES_first_second/train_first_second.py for randomized adversarial training based on TRADES.  

# Evaluation
PGD and CW evaluation with epsilon=0.031. Auto attack evaluation is under standard version with epsilon=8/255.  
We got the best performance between epoch 100-200.  

# Reference Code
[1] AT: https://github.com/locuslab/robust_overfitting  
[2] TRADES: https://github.com/yaodongyu/TRADES/  
[3] AutoAttack: https://github.com/fra31/auto-attack  
[4] MART: https://github.com/YisenWang/MART  
[5] AWP: https://github.com/csdongxian/AWP  
[6] AVMixup: https://github.com/hirokiadachi/Adversarial-vertex-mixup-pytorch  
[7] S2O: https://github.com/Alexkael/S2O

# Citation
**If you find our paper and repo useful, please cite our paper:**  
@article{jin2023randomized,  
  title={Randomized Adversarial Training via Taylor Expansion},  
  author={Jin, Gaojie and Yi, Xinping and Wu, Dengyu and Mu, Ronghui and Huang, Xiaowei},  
  journal={arXiv preprint arXiv:2303.10653},  
  year={2023}  
}
