# %%
import numpy as np
from collections import Counter

def alea(a, b, c, rho):
    # roh recommended between 100 and 1000
    sol = []
    for _ in range(rho):
        m = melange(b, c)
        com = complementarite(m, a, "", [])
        sol = sol + com 
    return sol


def melange(b, c):
    if len(b) == 0: return c
    n = np.random.randint(0, len(b)+1)
    return b[:n] + melange(c, b[n:])


def complementarite(m, a, r, s):
    sol = [x for x in s]
    
    if len(m) == 0:
        if len(a) == 0:
            sol.append(r)
    else:
        sol = complementarite(m[1:], a, r + m[0], sol)
        
        if len(a) == 0: return sol
        
        if m[0] == a[0]:
            sol = complementarite(m[1:], a[1:], r, sol)

    return sol






###############################################################################
###############################################################################



def classification(A, B, C, D, rho=10, n=1):
    results = alea(A, B, C, rho)
    return D in [x[0] for x in Counter(results).most_common(n)]

def classification_(A, B, C, D, n=1, rho=1000):
    results = alea(A, B, C, rho)
    l = [x[0] for x in Counter(results).most_common(n)]
    if D in l:
        return True, l.index(D)
    else:
        return False, 99999999


def classification_with_results(A, B, C, D, n=1, rho=1000):
    results = alea(A, B, C, rho)
    l = [x[0] for x in Counter(results).most_common(n)]
    if D in l:
        return True, l.index(D), l
    else:
        return False, 99999999, l

# %%
