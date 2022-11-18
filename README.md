# ebpy: Empirical Bayes in Python

Nonparametric Maximum Likelihood Estimator (NPMLE) for estimating Gaussian location mixture densities in d-dimensions from independent, potentially heteroscedastic observations. 

Basic usage:
<pre><code>from ebpy import GLMixture

m = GLMixture()

## Compute the NPMLE 
m.fit(X, prec)

## Denoised estimates based on empirical prior
gmleb = m.posterior_mean(X, prec) 
</code></pre>

![image1](demo/circle_demo.png)
