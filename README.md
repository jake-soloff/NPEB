# ebpy: Nonparametric Empirical Bayes in Python

<pre><code>
from ebpy import GLMixture

m = GLMixture()

## Compute the NPMLE 
m.fit(X, prec)

## Denoised estimates based on empirical prior
gmleb = m.posterior_mean(X, prec) 
</code></pre>