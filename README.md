# ISD: Interpolation-based Semi-supervised learning for object Detection (CVPR 2021)

By [Jisoo Jeong](http://mipal.snu.ac.kr/index.php/Jisoo_Jeong), [Vikas Verma](https://scholar.google.co.kr/citations?user=wo_M4uQAAAAJ&hl=en&oi=ao), [Minsung Hyun](https://scholar.google.com/citations?user=MpsUp10AAAAJ&hl=ko&oi=ao), [Juho Kannala](https://users.aalto.fi/~kannalj1/), [Nojun Kwak](http://mipal.snu.ac.kr/index.php/Nojun_Kwak)


#### For more details, please refer to our [arXiv paper](https://arxiv.org/abs/2006.02158)


## Installation & Preparation
We experimented with ISD using the SSD pytorch framework. To use our model, complete the installation & preparation on the [SSD pytorch homepage](https://github.com/amdegroot/ssd.pytorch)

#### prerequisites
- Python 3.6
- Pytorch 1.5.0

## Supervised learning
```Shell
python train_ssd.py
```

## CSD training
```Shell
python train_csd.py
```

## ISD training
```Shell
python train_isd.py
```

## Evaluation
```Shell
python eval.py
```
