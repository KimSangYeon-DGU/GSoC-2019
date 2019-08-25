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
We conducted researches to find out the stengths and weaknesses QGMM has.
### 1. Interference phenomena
According to paper, <img src="https://latex.codecogs.com/gif.latex?cos(\phi)" title="cos(\phi)" /> has an effect on the mixture case and QGMM and GMM are the same when <img src="https://latex.codecogs.com/gif.latex?\phi=\pi/2" title="\phi=\pi/2" />. Therefore, we checked its interference phenomena by visualizing it in 3D plotting.
<p align="center">
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/interferences.png">
</p>

From the above figures, we can check the interference phenomena as <img src="https://latex.codecogs.com/gif.latex?\phi" title="\phi" /> changed.  In addition, we can see when <img src="https://latex.codecogs.com/gif.latex?\phi=90" title="\phi=90" />, QGMM is the same with GMM.

### 2. Validity of the objective function
In the original paper, the objective function is that 

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;O(\theta_{k})=\sum_{i}&space;\sum_{k}Q_{i}(k)\log{P(p_{i},k|\theta_{k})}" title="O(\theta_{k})=\sum_{i} \sum_{k}Q_{i}(k)\log{P(p_{i},k|\theta_{k})}" />
</p>

In addition, the objective function means the expectation of the complete-data log likelihood, and we'll call it as log likelihood in this report.
However, the derivation of the covariance in the original paper has an error because Q shouldn't have an effect on the calculation, so we couldn't use it. Thus, we newly defined the objective function as an indicator of the training states.

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

### 3. Lambda impact
From the validity of the objective function research, we figured out it works properly. In addition, the higher value means the optimization is more constrained. Therefore, in this research, we checked the impact of lambda. Generally, the initial lambda can be calculated by NLL / approximation constraint from the objective function, but when the intial Gaussians are almost zero, we can't calculate NLL. Therefore, we set the initial value of lambda manually.

<p align="center">
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/05_impact_90_100.gif" width=256>
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/05_impact_90_1000.gif" width=256>
</p>
<p align="center">
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/05_impact_90_100_constraint.png" width=256>
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/05_impact_90_1000_constraint.png" width=256>
</p>

The above figures are the training process and the graph of the constraint. The left is with lambda 100 and the right is with lambda 1,000. From that, we found out that with lambda 100, the constraint was unstable and there are some cases in which the training works with the more-constrained optimization. However, we also found out that the too high lambda rather interferes with the convergence of the objective function.

### 4. Phi modeling
According to the original paper, <img src="https://latex.codecogs.com/gif.latex?cos(\phi)" title="cos(\phi)" /> can be calculated from that

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;cos(\phi)=\frac{1-\alpha_{1}^{2}-\alpha_{2}^{2}}{2\alpha_{1}\alpha_{2}\sum_{i}G_{i,1}G_{i,2}}" title="cos(\phi)=\frac{1-\alpha_{1}^{2}-\alpha_{2}^{2}}{2\alpha_{1}\alpha_{2}\sum_{i}G_{i,1}G_{i,2}}" />
</p>

However, when the initial Gaussians are almost zero, the <img src="https://latex.codecogs.com/gif.latex?cos(\phi)" title="cos(\phi)" /> is too large, exceeding the bound, <img src="https://latex.codecogs.com/gif.latex?\dpi{110}&space;-1\leq&space;cos(\phi)&space;\leq&space;1" title="-1\leq cos(\phi) \leq 1" />, and it results in the unstable training process. Therefore, we changed it to a trainable variable and the results in this final document were made after changing it. As the original paper mentioned, the phi difference is calculated from

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\phi_{l,k}=\phi_{k}-\phi_{l}" title="\phi_{l,k}=\phi_{k}-\phi_{l}" />
</p>

Thus, we checked the training results with the different initial values of phi.

<p align="center">
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/03_phi_0_1500.gif" width=256>
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/03_phi_90_1500.gif" width=256>
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/03_phi_180_1500.gif" width=256>
</p>
<p align="center">
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/03_phi_0_1500_phi.png" width=256>
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/03_phi_90_1500_phi.png" width=256>
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/03_phi_180_1500_phi.png" width=256>
</p>

In the above figures, the left, center, and right are with the initial values of phi 0 (45 - 45), 90 (45 - (-45)), and 180 (90 - (-90)) respectively. When we set the initial phi as 0, the values didn't changed, whereas in the cases of phi 90 and 180, they were changed. From some experiments, we found out that the two distributions get father as <img src="https://latex.codecogs.com/gif.latex?cos(\phi)" title="cos(\phi)" /> is positive, while they get closer as <img src="https://latex.codecogs.com/gif.latex?cos(\phi)" title="cos(\phi)" /> is negative.

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
In this research, we compared QGMM with GMM. As the indicator of the training performance, we use the percentage of the convergence on the clusters of the observations. We conducted 100 experiments with different initial means and the initial means were randomly generated between -1 and 1 from the maximum and minimum of x coordinates of the data set, and between -10 and 10 from the maximum and minimum of y coordinates of the data set. Besides, we used the augmented Lagrangian multiplier method for the constrained optimization. Like other researches, we didn't use any initial clustering like K-means.

<p align=center>
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/Convg1.png" width=400>
</p>

From the table above, there are 4 and 20 failed cases for QGMM with initial phi 0 and 90 respectively and 6 failed cases for GMM. Especially, among failed cases, there is a case that the training doesn't work, so we ran it again with other hyperparameters.

<p align="center">
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/TC1_failed_01.gif" width=256>
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/TC1_follow-up_01.gif" width=256>
</p>

In the images above, the left is one of the failed cases of QGMM with initial phi 0, and the right is followed up by changing its hyperparameter.

We conducted another 100 experiments. In this time, the initial means were randomly generated between -0.5 and 0.5 from the maximum and minimum of x coordinates of the data set, and between -5 and 5 from the maximum and minimum of y coordinates of the data set.

<p align=center>
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/Convg2.png" width=400>
</p>

From the above table, we can see the performance of QGMM with initial phi 0 increased a bit. We also checked some cases by changing phi.

<p align="center">
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/TC2_overlay_01.gif" width=256>
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/TC2_follow-up_01.gif" width=256>
</p>

The above images is the case that the two distributions are overlaid initially. The left is with initial phi 0, and the right is with initial phi 180. From the images, we can see that in left image, the two distributions moved to the same cluster first before moving to each cluster, on the other hand, the two distributions moved to each cluster. Therefore, we checked again that phi has an effect on the training process.

### 7. Multiple clusters
In this research, we checked the performance of QGMM in multiple clusters.


## Conclusions
We looked into the property of QGMM in this project and made some improvements in the training performance.  


## Contributions



## Acknowledgement
