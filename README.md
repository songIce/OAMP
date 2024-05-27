# OAMP
Matlab code for Structured PCA with Orthogonal Approximate Message Passing(OAMP).

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

### BAMP and Replica
* Refer to [Structured-PCA-](https://github.com/fcamilli95/Structured-PCA-)

