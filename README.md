<p align="center">
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/GSoC_logo.png" width=540>
</p>

## Title
Quantum Gaussian Mixture Models

## Organization
[mlpack](https://www.mlpack.org/)

## Mentor
[Sumedh Ghaisas](https://github.com/sumedhghaisas)

## Abstract
Gaussian Mixture Model (GMM) is widely used in computer vision as a state-of-the-art clustering algorithm. This project proposes Quantum Gaussian Mixture Model (QGMM) for Quantum Clustering and it is originally proposed in the [paper](https://arxiv.org/pdf/1612.09199.pdf). In this project, we implemented QGMM and conducted some experiments to see if how fast it trains, how better it models the data, what edge cases there are, and there is anything we can improve.

## Researches
We conducted researches to find out the strengths and weaknesses QGMM has.
### 1. Interference phenomena
According to the original paper, the probability equation of QGMM is that

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?P(p_{i},k|\theta_{k})&space;=&space;\alpha_{k}G_{i,k}\left&space;(&space;\sum_{l=1}^{K}&space;\alpha_{l}\cos(\phi_{l,k})G_{i,l}&space;\right&space;)" title="P(p_{i},k|\theta_{k}) = \alpha_{k}G_{i,k}\left ( \sum_{l=1}^{K} \alpha_{l}\cos(\phi_{l,k})G_{i,l} \right )" />
</p>
<p align="center">
  where <img src="https://latex.codecogs.com/gif.latex?G_{i,k}^{2}" title="G_{i,k}^{2}" /> is normalized
</p>

In addition, <img src="https://latex.codecogs.com/gif.latex?cos(\phi)" title="cos(\phi)" /> has an effect on the mixture case and QGMM and GMM are the same when <img src="https://latex.codecogs.com/gif.latex?\phi=\pi/2" title="\phi=\pi/2" />. Therefore, we checked its interference phenomena by visualizing it in 3D plotting.

<p align="center">
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/interferences.png">
</p>

From the above images, we can check the interference phenomena as <img src="https://latex.codecogs.com/gif.latex?\phi" title="\phi" /> changed.  In addition, we can see when <img src="https://latex.codecogs.com/gif.latex?\phi=90" title="\phi=90" />, QGMM is the same with GMM.

### 2. Validity of the objective function
In the original paper, the objective function is that 

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?\dpi{110}&space;O(\theta_{k})=\sum_{i}&space;\sum_{k}Q_{i}(k)\log{P(p_{i},k|\theta_{k})}" title="O(\theta_{k})=\sum_{i} \sum_{k}Q_{i}(k)\log{P(p_{i},k|\theta_{k})}" />
</p>

In addition, the objective function means the expectation of the complete-data log likelihood, and we'll call it as log likelihood in this report.
However, in the probability equation, because <img src="https://latex.codecogs.com/gif.latex?G_{i,k}" title="G_{i,k}" /> is unnormalized, we can't guarantee <img src="https://latex.codecogs.com/gif.latex?\inline&space;\sum_{i}\sum_{k}P(p_{i},k|\theta_{k})=1" title="\sum_{i}\sum_{k}P(p_{i},k|\theta_{k})=1" />. Thus, we newly defined the objective function as an indicator of the training status.

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?O(\theta_{k})=\left[-\sum_{i}&space;\sum_{k}&space;Q_{i}(k)\log{P(p_{i},k|\theta_{k}})\right]&plus;\lambda&space;\left[&space;\sum_{i}&space;\sum_{k}\{P(p_{i},k|\theta_{k})\}-1&space;\right]" title="O(\theta_{k})=\left[-\sum_{i} \sum_{k} Q_{i}(k)\log{P(p_{i},k|\theta_{k}})\right]+\lambda \left[ \sum_{i} \sum_{k}\{P(p_{i},k|\theta_{k})\}-1 \right]" />
</p>

Because <img src="https://latex.codecogs.com/gif.latex?G_{i,k}" title="G_{i,k}" /> is an unnormalized Gaussian in QGMM, we defined the new objective function like Lagrangian multiplier for constraint optimization. Therefore, the new objective function is <b><i>Negative Log Likelihood (NLL) + <img src="https://latex.codecogs.com/gif.latex?\lambda" title="\lambda" /> * Approximation Constraint</i></b> and using an optimizer, we'll minimize it. With the objective function, we conduct several experiments to check if it works properly.

<p align="center">
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/03_validity_90_1.gif" width=256>
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/04_validity_90_1.gif" width=256>
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/05_validity_90_1.gif" width=256>
</p>

From the above images, we can see the training works properly except for the right one (In the next research, we'll dig into the failed case).

### 3. Lambda(<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\lambda" title="\lambda" />) impact
From the validity of the objective function research, we figured out it works properly. In addition, the higher value means the optimization is more constrained. Therefore, in this research, we checked the impact of <img src="https://latex.codecogs.com/gif.latex?\lambda" title="\lambda" />. Generally, the initial <img src="https://latex.codecogs.com/gif.latex?\lambda" title="\lambda" /> can be calculated by NLL / approximation constraint from the objective function, but when the initial <img src="https://latex.codecogs.com/gif.latex?G_{i,k}" title="G_{i,k}" /> are almost zero, we can't calculate NLL. Therefore, we set the initial value of <img src="https://latex.codecogs.com/gif.latex?\lambda" title="\lambda" /> manually.

<p align="center">
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/05_impact_90_100.gif" width=256>
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/05_impact_90_1000.gif" width=256>
</p>
<p align="center">
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/05_impact_90_100_constraint.png" width=256>
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/05_impact_90_1000_constraint.png" width=256>
</p>

The above images are the training process and the graph of the constraint. The left is with <img src="https://latex.codecogs.com/gif.latex?\lambda" title="\lambda" /> 100 and the right is with <img src="https://latex.codecogs.com/gif.latex?\lambda" title="\lambda" /> 1,000. From that, we found out that with <img src="https://latex.codecogs.com/gif.latex?\lambda" title="\lambda" /> 100, the constraint was unstable and there are some cases in which the training works only when using the more-constrained optimization. However, the high <img src="https://latex.codecogs.com/gif.latex?\lambda" title="\lambda" /> is not always desirable because we also found out that the too high <img src="https://latex.codecogs.com/gif.latex?\lambda" title="\lambda" /> rather interferes with the convergence of the objective function.

### 4. Phi(<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\phi" title="\phi" />) modeling
According to the original paper, <img src="https://latex.codecogs.com/gif.latex?cos(\phi)" title="cos(\phi)" /> can be calculated from that

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?\dpi{110}&space;cos(\phi)=\frac{1-\alpha_{1}^{2}-\alpha_{2}^{2}}{2\alpha_{1}\alpha_{2}\sum_{i}G_{i,1}G_{i,2}}" title="cos(\phi)=\frac{1-\alpha_{1}^{2}-\alpha_{2}^{2}}{2\alpha_{1}\alpha_{2}\sum_{i}G_{i,1}G_{i,2}}" />
</p>

However, when the initial <img src="https://latex.codecogs.com/gif.latex?G_{i,k}" title="G_{i,k}" /> are almost zero, the <img src="https://latex.codecogs.com/gif.latex?cos(\phi)" title="cos(\phi)" /> is too large, exceeding the bound, <img src="https://latex.codecogs.com/gif.latex?\dpi{110}&space;-1\leq&space;cos(\phi)&space;\leq&space;1" title="-1\leq cos(\phi) \leq 1" />, and it results in the unstable training process. Therefore, we changed it to a trainable variable and the results in this final document were made after changing it. As the original paper mentioned, the <img src="https://latex.codecogs.com/gif.latex?\phi" title="\phi" /> difference is calculated from

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?\dpi{110}&space;\phi_{l,k}=\phi_{k}-\phi_{l}" title="\phi_{l,k}=\phi_{k}-\phi_{l}" />
</p>

Thus, we checked the training results with the different initial values of <img src="https://latex.codecogs.com/gif.latex?\phi" title="\phi" />.

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

In the above images, the left, center, and right are with the initial values of <img src="https://latex.codecogs.com/gif.latex?\phi" title="\phi" /> 0 (45 - 45), 90 (45 - (-45)), and 180 (90 - (-90)) respectively. When we set the initial <img src="https://latex.codecogs.com/gif.latex?\phi" title="\phi" /> as 0, the values weren't changed, whereas in the cases of <img src="https://latex.codecogs.com/gif.latex?\phi" title="\phi" /> 90 and 180, they were changed. From some experiments, we found out that the two distributions get father as <img src="https://latex.codecogs.com/gif.latex?cos(\phi)" title="cos(\phi)" /> is positive, while they get closer as <img src="https://latex.codecogs.com/gif.latex?cos(\phi)" title="cos(\phi)" /> is negative.

### 5. Mixed clusters
Using mlpack's GMM class, we generated the data set for the mixed clusters to see if how QGMM works. To generate the mixture, we drew a circle between the two clusters and generated observations randomly.

<p align="center">
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/Mixed data set.png" width=640>
</p>

Using the above data sets, we trained QGMM and GMM. Especially, we investigated two cases for QGMM with the initial <img src="https://latex.codecogs.com/gif.latex?\phi" title="\phi" /> 0 and 90 to check the impact of the initial <img src="https://latex.codecogs.com/gif.latex?\phi" title="\phi" />.

<p align="center">
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/Mixed results.png" width=640>
</p>

From the above results, we found out the results between QGMM and GMM are totally different. Furthermore, even between QGMMs, the results vary depending on <img src="https://latex.codecogs.com/gif.latex?\phi" title="\phi" />. 

### 6. Comparison with GMM
In this research, we compared QGMM with GMM. As the indicator of the training performance, we use the percentage of the convergence on the clusters of the observations. We conducted 100 experiments with different initial means and the initial means were randomly generated between -1 and 1 from the maximum and minimum of x coordinates of the data set, and between -10 and 10 from the maximum and minimum of y coordinates of the data set. Besides, we used the augmented Lagrangian multiplier method for constrained optimization. Like other researches, we didn't use any initial clustering like K-means.

<p align=center>
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/Convg_01.png" width=400>
</p>

From the table above, there are 4 and 20 failed cases for QGMM with initial <img src="https://latex.codecogs.com/gif.latex?\phi" title="\phi" /> 0 and 90 respectively and 6 failed cases for GMM. Especially, among failed cases, there is a case that the training doesn't work, so we ran it again with other hyperparameters.

<p align="center">
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/TC1_failed_01.gif" width=256>
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/TC1_follow-up_01.gif" width=256>
</p>

In the images above, the left is one of the failed cases of QGMM with initial <img src="https://latex.codecogs.com/gif.latex?\phi" title="\phi" /> 0, and the right is followed up by changing its hyperparameter.

We conducted another 100 experiments. In this time, the initial means were randomly generated between -0.5 and 0.5 from the maximum and minimum of x coordinates of the data set, and between -5 and 5 from the maximum and minimum of y coordinates of the data set.

<p align=center>
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/Convg_02.png" width=400>
</p>

From the above table, we can see the performance of QGMM with initial <img src="https://latex.codecogs.com/gif.latex?\phi" title="\phi" /> 0 increased a bit. We also checked some cases by changing <img src="https://latex.codecogs.com/gif.latex?\phi" title="\phi" />.

<p align="center">
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/TC2_overlay_01.gif" width=256>
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/TC2_follow-up_01.gif" width=256>
</p>

The above images are the case that the two distributions are overlaid initially. The left is with initial <img src="https://latex.codecogs.com/gif.latex?\phi" title="\phi" /> 0, and the right is with initial <img src="https://latex.codecogs.com/gif.latex?\phi" title="\phi" /> 180. From the images, we can see that in the left image, the two distributions moved to the same cluster first before moving to each cluster, on the other hand, the two distributions moved to each cluster. Therefore, we checked again that <img src="https://latex.codecogs.com/gif.latex?\phi" title="\phi" /> has an effect on the training process.

### 7. Multiple clusters
In this research, we checked the performance of QGMM in multiple clusters.

<p align="center">
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/multiple_01.gif" width=256>
  <img src="https://github.com/KimSangYeon-DGU/GSoC-2019/blob/master/images/multiple_02.gif" width=256>
</p>

In the images above, the left is with the sequence of <img src="https://latex.codecogs.com/gif.latex?\phi_{k}" title="\phi_{k}" />, [0, 0, 0, 0, 0] and the right is with the sequence of <img src="https://latex.codecogs.com/gif.latex?\phi_{k}" title="\phi_{k}" />, [45, -45, 45, -45, 45]. For the multiple clusters cases, it's tricky to set the initial sequence of <img src="https://latex.codecogs.com/gif.latex?\phi_{k}" title="\phi_{k}" /> to model the data properly.

## Conclusions
In this project, we found some errors in the original QGMM, tried to correct them, and made some improvements in its performance while we looked into the property of it through the various trials. Before implementing QGMM, we simply visualized and checked the 3D probability space of QGMM to investigate its impact and come up with methods to normalize the probability to make the integral of it one.
  
Also, we found an error in the derivation of the covariance in the original approach, so we newly defined the objective function with the approximation constraint for the probability normalization, and checked it works properly. 

While looking into the training states, we found the value of the cosine of <img src="https://latex.codecogs.com/gif.latex?\phi" title="\phi" /> is too large, so we changed <img src="https://latex.codecogs.com/gif.latex?\phi" title="\phi" /> as a trainable variable. As a result, training performance became stable than before.

As we saw in the comparison with GMM research, QGMM showed flexible performance by adjusting the hyperparameters. In other words, we should set the proper hyperparameters to model the data correctly, but sometimes it would be not easy to do. Furthermore, from some several experiments, we found out that <img src="https://latex.codecogs.com/gif.latex?\phi" title="\phi" /> has a significant effect on the training process. In particular, it's hard to set the initial sequence of <img src="https://latex.codecogs.com/gif.latex?\phi" title="\phi" /> when more than 3 clusters cases. Therefore, the current QGMM needs to come up with how to control <img src="https://latex.codecogs.com/gif.latex?\phi" title="\phi" /> to generalize its performance.

## Blog
- [mlpack GSoC blog](https://www.mlpack.org/gsocblog/SangyeonKimPage.html)

## Contributions
- [Convert DiagonalGMMs into GMMs in `mlpack_gmm_train`](https://github.com/mlpack/mlpack/pull/1860)
- [Make `GradientTransposedConvolutionLayerTest` robust.](https://github.com/mlpack/mlpack/pull/1777)
- [Synchronize parameter's name with @param in `dists` and `gmm` directory.](https://github.com/mlpack/mlpack/pull/1753)
- [Resolve gcc parameter reorder warning for clean build.](https://github.com/mlpack/mlpack/pull/1738)
- [Edited to comply with the Style Checks](https://github.com/mlpack/mlpack/pull/1713)
- [Implement wrapper for diagonally-constrained GMM HMMs.](https://github.com/mlpack/mlpack/pull/1666)
- [Changed the file name `acrobat.hpp` to `acrobot.hpp`.](https://github.com/mlpack/mlpack/pull/1654)
- [Edited mountain_car.hpp to bound to goal position.](https://github.com/mlpack/mlpack/pull/1649)
- [Fix/Change Windows installer](https://github.com/mlpack/mlpack/pull/1642)
- [Fixed DBSCAN isn't using PointSelectionPolicy issue ](https://github.com/mlpack/mlpack/pull/1627)

## Acknowledgement
Massive thanks Sumedh. He gave me great support and guidance for the project. Whenever I got stuck with problems, he presented possible solutions with enough description. While doing this project with him, I got many impressions from his inventive ideas and insightful approaches to the researches and learned a lot from him. Lastly, there are many proficient developers and researchers in mlpack community. It's my pleasure to contribute to this great machine learning library and I'll continue to contribute to mlpack actively. Thanks again Sumedh, and everyone. :)
