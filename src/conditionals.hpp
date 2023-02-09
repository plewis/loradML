// Code wrapped in GHM conditional compiles was introduced in order to compute the
// Generalized Harmonic Mean estimator described here:
//
// S Arima and L Tardella. 2012. Improved harmonic mean estimator for
// phylogenetic model evidence. Journal of Computational Biology 19:418-438.
//
// This code was used to produce numbers needed for Table 1 in this paper:
//
// YB Wang, A Milkey, A Li, MH Chen, L Kuo, and PO Lewis. LoRaD: marginal
// likelihood estimation with haste (but no waste). Systematic Biology (in revision)
//
// It is not general and this define should not be uncommented unless your goal is
// specifically to reproduce the GHM analyses in the above Wang et al. paper.
#define GHM
