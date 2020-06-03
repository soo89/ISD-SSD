# CSD: Consistency-based Semi-supervised learning for object Detection

By [Jisoo Jeong](http://mipal.snu.ac.kr/index.php/Jisoo_Jeong), [Vikas Verma](https://scholar.google.co.kr/citations?user=wo_M4uQAAAAJ&hl=en&oi=ao), [Minsung Hyun](http://mipal.snu.ac.kr/index.php/MinSung_Hyun), [Nojun Kwak](http://mipal.snu.ac.kr/index.php/Nojun_Kwak)



## Installation & Preparation
We experimented with ISD using the SSD pytorch framework. To use our model, complete the installation & preparation on the [SSD pytorch homepage](https://github.com/amdegroot/ssd.pytorch)

#### prerequisites
- Python 3.6
- Pytorch 1.0.0

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
