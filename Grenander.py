import numpy as np
from scipy.interpolate import interp1d
from sklearn.isotonic import isotonic_regression

class Grenander:
    """Grenander estimator of a monotone decreasing density.

    Parameters
    ----------

    x_min : float, default=0
        Lower bound on the domain. 
    
    x_max : float or None, default=float('Inf')
        Upper bound on the domain. If None, the upper bound 
        on the domain is the max of the data supplied to `fit`.

    Attributes
    ----------

    cdf : function
        accepts data and returns the estimated distribution function

    pdf : function
        accepts data and returns the estimated probability density function

    knots : array-like of shape (k,)
        knots of the least concave majorant of the empirical cdf

    heights : array-like of shape (k,)
        heights of the least concave majorant of the empirical cdf
        corresponding to the knots
        
    slopes : array-like of shape (k-1,)
        left-hand slopes of the least concave majorant of the empirical cdf
        corresponding to knots[1:]

    """
    def __init__(self, x_min=0, x_max=float('Inf')):
        self.x_min = x_min
        self.x_max = x_max

    def fit(self, x):
        """Compute the Grenander estimator using isotonic regression.

        Parameters
        ----------

        x : array-like of shape (n,)
            observed data

        Returns
        -------

        fitted_values : array-like of shape (n,)
            the value of the Grenander estimator at x
        
        """
        n = len(x)
        
        I = x.argsort()
        x_= x[I]

        # domain of empirical cdf of x
        W = np.hstack([self.x_min, x_])

        # compute slopes of LCM of empirical cdf
        # NB: these are left-hand slopes, so no slope is defined at x_min
        w = W[1:] - W[:-1]
        slopes = isotonic_regression(np.ones(n)/(n*w), 
                                     sample_weight=w.copy(), 
                                     increasing=False)
        fitted_values = slopes[I.argsort()]
        
        # save only the knots for faster computation
        keep = np.ones(n, dtype=bool)
        keep[:-1] = ~np.isclose(slopes[:-1], slopes[1:])

        # compute LCM of empirical cdf
        slopes_ = slopes[keep]
        knots_  = np.hstack([self.x_min, x_[keep]])
        heights_= np.hstack([0, (np.where(keep)[0]+1)/n])

        x_max = self.x_max
        if x_max is not None:
            slopes_ = np.hstack([slopes_, 0])
            knots_ = np.hstack([knots_, x_max])
            heights_ = np.hstack([heights_, 1])

        self.slopes = slopes_
        self.knots = knots_
        self.heights = heights_

        self.cdf = interp1d(self.knots, 
                            self.heights, 
                            kind='linear',
                            assume_sorted=True)
        self.pdf = interp1d(self.knots, 
                            np.hstack([np.nan, self.slopes]), 
                            kind='next',
                            assume_sorted=True)

        # return fitted values in original order
        return fitted_values

def lfdr(p, zeta=None): 
    """Estimate the local false discovery rate using the Grenander estimator.

    Parameters
    ----------

    p : array-like of shape (n,)
        observed p-values

    zeta : float or None
        threshold used for estimating null proportion pi0
        if None, the estimate pi0=1 is used

    Returns
    -------

    l : array-like of shape (n,)
        the estimated l-values
    
    fdr : function
        maps a p-value to its estimated lfdr

    """
    if zeta is not None:
        pi0 = (np.sum(p>zeta)+1)/((1-zeta)*len(p))
    else:
        pi0 = 1

    gren = Grenander(x_max=1)
    fhat = gren.fit(p)

    return(pi0/fhat, lambda t: pi0/gren.pdf(t))

if __name__=="__main__":
    import matplotlib.pyplot as plt

    n = 50
    x = np.random.beta(.4, 1, n)
    m = Grenander(x_max=1)

    slopes = m.fit(x)

    plt.figure(figsize=(8,8))
    plt.plot(np.sort(x), np.arange(1,n+1)/n, 'b.')
    plt.plot(m.knots, m.heights, 'r.-')
    xx = np.linspace(0, 1, 1000)
    plt.plot(xx, m.cdf(xx), 'g--')

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')

    plt.ylim([0, 1.01])
    plt.xlim([0, 1])
    plt.show()

    plt.plot(x, slopes, 'b.')
    plt.plot(xx, m.pdf(xx), 'b.')
    plt.ylim([-.01, 2])
    plt.xlim([0, 1])

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')

    plt.show()

    n = 2000
    pi0 = .75
    p = np.random.rand(n)
    a, b = .3, 1
    p[1500:] = np.random.beta(a, b, 500)

    _, l = lfdr(p, zeta=.5)

    from scipy import stats

    tt = np.linspace(0, 1, 10000)
    plt.plot(tt, pi0/(pi0+(1-pi0)*stats.beta.pdf(tt, a, b)), 'k-')
    plt.plot(tt, l(tt), 'b--')

    plt.ylim([-.01, .5])
    plt.xlim([0, .01])

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')

    plt.show()