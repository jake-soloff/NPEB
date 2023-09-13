# npeb: Nonparametric Empirical Bayes in Python

Nonparametric Maximum Likelihood Estimator (NPMLE) for estimating Gaussian location mixture densities in d-dimensions from independent, potentially heteroscedastic observations. 

## Installation

The easiest way to install npeb is using ``pip``:
<pre><code> pip install npeb </code></pre>

## Basic usage
<pre><code>from npeb import GLMixture

m = GLMixture()

## Compute the NPMLE 
m.fit(X, prec)

## Denoised estimates based on empirical prior
gmleb = m.posterior_mean(X, prec) 
</code></pre>

![image1](demo/circle_demo.png)
