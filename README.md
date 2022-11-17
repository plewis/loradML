# loradML

This software estimates the Bayesian marginal likelihood from a parameter sample using the LoRaD method as described in this paper:

> Y-B Wang, A Milkey, A Li, M-H Chen, L Kuo, and PO lewis. 2023. LoRaD: marginal likelihood estimation with haste (but no waste). Systematic Biology (in revision).

## Tutorial

loradML looks for a file named _loradml.conf_ in the current directory and obtains run-time settings from that file. While it is possible to specify everything on the command line, you will find that using a _loradml.conf__ file is much more convenient due to the number of `colspec` entries that need to be specified.

In this tutorial you will produce a posterior sample using [RevBayes](https://revbayes.github.io) and analyze it with loradML. The files needed are in the directory _test_:

* _green5.nex_ is a NEXUS-formatted data file containing 5 rbcL sequences from diverse green plants
* _green5.tre_ is a NEXUS tree file containing the maximum likelihood tree for the 5 sequences under a GTR+I+G model
* _green5.Rev_ is the RevBayes script used to generate a posterior sample
* _loradml.conf_ is the settings file used by loradML

Using RevBayes is not a focus of this tutorial, but, briefly, the _green5.Rev_ file will carry out a Bayesian MCMC simulation under the GTR+I+G model for 100,000 generations (after 1000 generations burn-in), saving samples to the file *posterior_samples.log* every 100 generations. It is important to pay attention to the parameters defined in the model. These can be identified in the _green5.Rev_ file by lines that contain a tilde (~) symbol, as these are the lines in which priors are defined for the parameters. Here are all such lines in the _green5.Rev_ file:

```
tree_length ~ dnGamma(shape=1, rate=0.1)
edge_length_proportions ~ dnDirichlet( rep(1.0,num_branches) )
freq ~ dnDirichlet(v(1,1,1,1))
er ~ dnDirichlet(v(1,1,1,1,1,1))
ratevar ~ dnExponential(1.0)
pinvar ~ dnBeta(1.0, 1.0)
seq  ~ dnPhyloCTMC(tree=phylogeny, Q=Qmatrix, type="DNA", siteRates=site_rates, pInv=pinvar)
```

The only one of these lines we will ignore is the `seq ~ dnPhyloCTMC(...)` line because this specifies the distribution of the data conditional on the parameters (i.e. the likelihood function) and does not represent a prior specification.

How many parameters are there?

No. free parameters | Parameter variable
:-----------------: | :-------------------:
         1          | tree_length
         6          | edge_length_proportions
         3          | freq
         5          | er
         1          | ratevar
         1          | pinvar
         
In total, there are 17 free parameters in this model. Note that there are 7 edges in the ML tree but we only count 6 edge length proportions because, being proportions, these are constrained to sum to 1. Likewise, we only count 3 and 5 free parameters for the 4 nucleotide relative frequencies and the 6 exchangeabilities, respectively. Tree topology does not enter into the equation because our RevBayes script assumes that the topology is fixed to the ML topology.

Run the _green5.Rev_ script in RevBayes to generate the *parameter_samples.log* file.

You will note that the *parameter_samples.log* file has more than 17 columns. We must therefore tell loradML about each column in the file, in particular identifying the iteration column, the column providing the log-posterior kernel (i.e. log-likelihood plus log-prior), and we need to tell it about the nature of all 17 free parameters.

Here is the _loradml.conf_ file for this example:

```
# Comments begin with a hash symbol
paramfile = posterior_samples.log
startsample = 2
trainingfrac = 0.5
coverage = 0.5

colspec=  iteration    Iteration	
colspec=  posterior    Posterior	
colspec=  ignore       Likelihood	
colspec=  ignore       Prior	
colspec=  ignore       alpha	
colspec=  simplex      edge_length_proportion	
colspec=  simplex      edge_length_proportion	
colspec=  simplex      edge_length_proportion	
colspec=  simplex      edge_length_proportion	
colspec=  simplex      edge_length_proportion	
colspec=  simplex      edge_length_proportion	
colspec=  simplexfinal edge_length_proportion	
colspec=  ignore       edgelen	
colspec=  ignore       edgelen	
colspec=  ignore       edgelen	
colspec=  ignore       edgelen	
colspec=  ignore       edgelen	
colspec=  ignore       edgelen	
colspec=  ignore       edgelen	
colspec=  simplex      exchangeability
colspec=  simplex      exchangeability
colspec=  simplex      exchangeability
colspec=  simplex      exchangeability
colspec=  simplex      exchangeability
colspec=  simplexfinal exchangeability	
colspec=  simplex      basefreq	
colspec=  simplex      basefreq	
colspec=  simplex      basefreq	
colspec=  simplexfinal basefreq	
colspec=  proportion   pinvar	
colspec=  positive     ratevar	
colspec=  ignore       site_rate	
colspec=  ignore       site_rate	
colspec=  ignore       site_rate	
colspec=  ignore       site_rate	
colspec=  positive     tree_length	
```

The first line is straightforward, specifying the name of the sample file to process as the setting `paramfile`. 

The `startsample` setting says to start with the second sample (the first sample in the file is just the starting state and is not a valid sample from the posterior distribution). There is a corresponding `endsample` that can be used to specify the last sample included. If `startsample` is not specified or is set to 0, the first sample included will be sample 1. If `endsample` is not specified or is set to 0, the last sample included will be the final sample in the file. 

I will talk about the settings `trainingfrac` and `coverage` after explaining the `colspec` entries.

The LoRaD method requires all parameters to be unconstrained (i.e. their support should be the entire real line), so we need to tell loradML what the support is for each parameter in the model so that it can transform those that are constrained (for example, tree length is constrained to the positive half of the real line). There are 36 `colspec` entries, one for each column in the `paramfile`. Each `colspec` entry comprises a column type and a label separated by whitespace. Note that there should be one `colspec` entry for every column and the order of `colspec` entries should match the ordering of the columns.

The column type should be one of these values:

column type   | description
:----------:  | :----------
  iteration   | this column stores the RevBayes generation in which the sample was taken
   ignore     | this column should be entirely ignored
 posterior    | all columns of this type should be added together to create the log of the unnormalized posterior density
 positive     | this parameter is strictly positive and should be log transformed
 simplex      | this column is part of a multivariate parameter that represents a point on a simplex; such parameters are log ratio transformed using the final component as the ratio denominator
simplexfinal  | this column is the last column of a multivariate simplex parameter; it is used as the reference in the log ratio transformation
unconstrained | this column represents a parameter that is already unconstrained and thus needs no transformation

Recall that we identified 17 parameters in the model used in this example. Now note that there are 17 `colspec` entries that have column type `positive`, `simplex`, or `unconstrained`. These correspond to columns 6-11, 20-24, 26-28, 30, 31, and 36. The columns having type `simplexfinal` do not count as parameters because they represent the degree of freedom lost due to the constraint that simplex parameter components must add to 1.0. Note also that only the column `Posterior` was given column type `posterior` (not `Likelihood` or `Prior`). This is because the `Posterior` is already the sum of the `Likelihood` and `Prior` columns. We could, alternatively and equivalently, have `ignore`d the `Posterior` column and specified `posterior` column type for both the `Likelihood` column and `Prior` column.

For all transformations that it carries out, loradML keeps track of the appropriate Jacobian factors. It also performs one further standardization transformation. This involves subtracting the mean vector from each sampled vector and multiplying by the inverse standard deviation matrix in order achieve something close to a multivariate normal posterior density surface.

The LoRaD method defines a _working parameter space_ using some fraction of the sample, and this fraction is specified by the _trainingfrac_ setting. Normally you should set this value to 0.5. That is, the first half of the sample will be used to determine the extent of the working parameter space, and only the second half will be used in estimating the marginal likelihood. The fraction 1 - `trainingfrac` is called the estimation fraction and the corresponding sample is the estimation sample.

On the training sample, LoRaD performs individual parameter transformations, estimates the mean vector and covariance matrix of the transformed samples, and then performs the standardization transformation. It then sorts the transformed and standardized samples in the training fraction according to their distance from the origin and uses the fraction `coverage` closest to the origin to determine the radius (termed rmax hereafter) of the working parameter space. This radius represents the "lowest radial distance" from which the LoRaD acronym derives.

The mean vector and covariance matrix calculated from the training fraction are used to standardize the transformed sample vectors in the estimation sample. Because the mean vector is determined from the training sample and not from the estimation sample, the estimation sample will not be exactly centered over the origin. The estimation sample is sorted from low to high distance (radius) from the origin and only estimation sample vectors that are closer to the origin than the cutoff value rmax (calculated from the training sample) are used.

You can easily modify the _green5.Rev_ script to conduct a Steppingstone analysis, which will return a log marginal likelihood very close to the value reported by loradML. To run steppingstone, change `do_mcmc = TRUE` to `do_mcmc = FALSE` at the top of the _green5.Rev_ file and rerun it.

## Running loradML

To run loradML, simply invoke it in the same directory as the _loradml.conf_ file and the RevBayes parameter file. Specifying `--quiet` or `-q` on the command line results in minimal output. Specifying `--mcse` or `-m` on the command line triggers estimation of the Monte Carlo Standard Error (MCSE) using an Overlapping Batch Statistics (OBS) approach. In this case, the setting `tbratio` is used to determine the batch size relative to the sample size; a `tbratio` of 10 to 20 is recommended. If this results in a batch size less than 200, the program will refuse to run. Note that OBS is computationally intensive because estimating the MCSE requires estimation of the marginal likelihood potentially many thousands of times (once for each overlapping batch).

Note that loradML can be used with virtually any Bayesian MCMC program. All that is required is that the program create a tab-delimited parameter sample file specifying the log posterior kernel (or the log-likelihood and log-prior) and containing a column for each parameter in the model.






