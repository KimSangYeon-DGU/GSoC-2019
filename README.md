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

### 2. Phi modeling

### 3. Validity of the objective function
In the original paper, the objective function is that 

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;O(\theta_{k})=\sum_{i}&space;\sum_{k}Q_{i}(k)\log{P(p_{i},k|\theta_{k})}" title="O(\theta_{k})=\sum_{i} \sum_{k}Q_{i}(k)\log{P(p_{i},k|\theta_{k})}" />
</p>

In addition, the objective function means the expectation of the complete-data log likelihood, and we'll call it as log likelihood in this report.
However, the derivation of the covariance in the original paper has an error because Q shouldn't have an effect on the calculation, so we couldn't use it. Thus, we newly defined the objective function as an indication of the training states.

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;O(\theta_{k})=-\sum_{i}&space;\sum_{k}[Q_{i}(k)\log{P(p_{i},k|{\theta_{k}})}]&plus;\lambda[\sum_{i}&space;\sum_{k}\{P(p_{i},k|\theta_{k})\}-1]" title="O(\theta_{k})=-\sum_{i} \sum_{k}[Q_{i}(k)\log{P(p_{i},k|{\theta_{k}})}]+\lambda[\sum_{i} \sum_{k}\{P(p_{i},k|\theta_{k})\}-1]" />
</p>

Because Gaussians are unnormalized in QGMM, we defined the new objectvie function like Lagrangian multiplier for constraint optimization. Therefore, the new objective function is NLL + lambda * approximation constraint and using an optimizer, we'll minimize it. With the objective function, we conduct several experiments to check if it works properly.

<p align="center">
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/03_validity_90_1.gif" width=256>
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/04_validity_90_1.gif" width=256>
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/05_validity_90_1.gif" width=256>
</p>

From the above figures, we can see the training works properly except for the right one (In the next research, we'll dig into the failed case).

### 4. Lambda impact
From the validity of the objective function research, we figured out it works properly. In addition, the higher value means the optimization is more constrained. Therefore, in this research, we checked the impact of lambda. Generally, the initial lambda can be calculated by NLL / approximation constraint from the objective function, but when the intial Gaussians are almost zero, we can't calculate NLL. Therefore, we set the initial value of lambda manually.

<p align="center">
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/05_impact_90_100.gif" width=256>
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/05_impact_90_1000.gif" width=256>
</p>
<p align="center">
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/05_impact_90_100_constraint.png" width=256>
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/05_impact_90_1000_constraint.png" width=256>
</p>

The above figures are the training process and the graph of the constraint. The left is with lambda 100 and the right is with lambda 1,000. From that, we found out that with lambda 100, the constraint was unstable and there are some cases in which the training works with the more-constrained optimization.

### 5. Mixed clusters
Using mlpack's GMM class, we generated the mixed clusters data set to see if how QGMM works. To generate the mixture, we drew a circle between the two clusters and generated observations randomly.

<p align="center">
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/Mixed data set.png" width=512>
</p>

Using the above data sets, we trained QGMM and GMM. Especially, there are two trainings for QGMM with the initial phi 0 and 90 to check the impact of the initial phi.

<p align="center">
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/Mixed results.png" width=512>
</p>

From the aboves results, we found out the results between QGMM and GMM are totally different. Furthermore, even between QGMMs, the results vary depending on <img src="https://latex.codecogs.com/gif.latex?\phi" title="\phi" />. 

### 6. Comparison with GMM
In this research, we did compare QGMM with GMM. As the indicator of the training performance, we use the percentage of the convergence on the clusters of the observations. 

<p align=center>
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/Convg1.png" width=400>
</p>

### 7. Multiple clusters

## Conclusions

## Contributions

## Acknowledgement
