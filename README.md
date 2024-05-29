# OAMP(Orthogonal Approximate Message Passing)
Matlab code for [Optimality of Approximate Message Passing Algorithms for Spiked Matrix Models with Rotationally Invariant Noise](https://arxiv.org/abs/2405.18081).

# References
The BAMP algorithms and Replica results are implemented base on the following papers
* [Bayes-optimal limits in structured PCA, and how to reach them](https://arxiv.org/pdf/2210.01237)
* [Structured-PCA-](https://github.com/fcamilli95/Structured-PCA-)

# Content
### OAMP 
* OAMP.m: Optimal orthogonal message passing algorithm with Resolvent estimator
* OAMP_SE_FP: Solves Fixed points of SE and calculates corresponding free energy
* OAMP_Poly: Orthogonal message passing algorithm with Polynomial estimator

### Universality
* OAMP_realcov_art_signal.m: Orthogonal message passing algorithm with Polynomial estimator for artificial signal and noise from real data.
* Real data are generated following the methodology described in [Empirical Bayes PCA in high dimensions](https://arxiv.org/abs/2012.11676)

### BAMP
BAMP code is based on [Structured-PCA-](https://github.com/fcamilli95/Structured-PCA-)
* BAMP_quartic.m/BAMP_sestic.m: Implements the BAMP algorithm for quartic/sestic potential with Rademacher prior, sparse Rademacher prior, and 2-Point prior.
* BAMP_SE_quartic.m/BAMP_SE_quartic_2Points: State evolution of BAMP with quartic potential for sparse Rademacher prior/2-Point prior.
* BAMP_SE_sestic.m/BAMP_SE_sestic_2Points: State evolution of BAMP with sestic potential for Rademacher prior/2-Point prior.

### Replica PCA
Replica code is based on [Structured-PCA-](https://github.com/fcamilli95/Structured-PCA-)

### data
density_freecum.m/density_freecum_power6.m: Generates free cumulants of quartic/sestic potential. This code is from [Structured-PCA-](https://github.com/fcamilli95/Structured-PCA-)


