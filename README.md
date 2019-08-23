# GSoC-2019

## Title
Quantum Gaussian Mixture Models

## Organization
[mlpack](https://www.mlpack.org/)

## Mentor
[Sumedh Ghaisas](https://github.com/sumedhghaisas)

## Abstract
Gaussian Mixture Model (GMM) is widely used in computer vision as a state-of-the-art clustering algorithm. This project proposes Quantum Gaussian Mixture Model (QGMM) for Quantum Clustering and it is originally proposed in the [paper](https://arxiv.org/pdf/1612.09199.pdf). In this project, we implemented QGMM and conducted some experiments to see if how fast it trains, how better it models the data, what edge cases there are, and there is anything we can improve.

## Researches
We conducted 7 researches to find out the stengths and weaknesses QGMM has.
### 1. Interference phenomena
According to paper, <img src="https://latex.codecogs.com/gif.latex?cos(\phi)" title="cos(\phi)" /> has an effect on the mixture case and QGMM and GMM are the same when <img src="https://latex.codecogs.com/gif.latex?\phi=\pi/2" title="\phi=\pi/2" />. Therefore, we checked its interference phenomena by visualizing it in 3D plotting.
<p align="center">
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/interferences.png">
</p>

From the above figures, we can check the interference phenomena as <img src="https://latex.codecogs.com/gif.latex?\phi" title="\phi" /> changed.  In addition, we can see when <img src="https://latex.codecogs.com/gif.latex?\phi=90" title="\phi=90" />, QGMM is the same with GMM.

### 2. Validity of the objective function
In the original paper, the derivation of the covariance has an error because Q shouldn't have an effect on the calculation. So, we newly defined the objective function as an indication of the training states.

### 3. Lambda impact
### 4. Phi modeling
### 5. Mixed clusters
### 6. Comparison with GMM
### 7. Multiple clusters

## Conclusions

## Contributions

## Acknowledgement
