from sklearn import isotonic

def cusum(x):
    return np.concatenate(([0], np.cumsum(x)))

def unsum(x):
    return x[1:] - x[:-1]

# given the point-set Pi = (Wi, Gi) s.t. W1 ≤...≤ Wn,
# return the sequence G* s.t. (Wi, G*_i) is the GCM of P
def GCM(W, G):
    w = unsum(W)
    g = unsum(G)/w
    g_= isotonic.isotonic_regression(g.copy(), sample_weight=w.copy())
    return cusum(g_*w)

def LCM(W, G):
    return -GCM(W, -G)

def grenander(x):
    n = len(x)
    W = np.hstack([0, np.sort(x)])
    G = np.linspace(0,1,n+1)
    G_= LCM(W, G)
    slopes = unsum(G_)/unsum(W)
    return slopes
