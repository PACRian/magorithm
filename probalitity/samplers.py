import random

import numpy as np


def get_sample(sampler, x0=None, y0=None, k=15):
    if x0 is not None and y0 is not None:
        sampler.send([x0, y0])
    for i, r in enumerate(sampler):
        if i==k:
            return r 

def get_trajection(sampler, x0=None, y0=None, k=15):
    if x0 is not None and y0 is not None: sampler.send([x0, y0])
    x, y = [], []
    for i, r in enumerate(sampler):
        print(r)
        x.append(r[0])
        y.append(r[1])
        if i==k:
            return x, y

def acce_rej_sampling(f_density, proposal_density, proposal_rand, sz=100, m=1, simple_return=True):
    u = np.random.rand(sz)
    x = proposal_rand(sz)
    y = m*u*proposal_density(x)
    _is_accept = y <= f_density(x)

    if simple_return:
        return x[_is_accept]
    else:
        return x[_is_accept], y[_is_accept], x[~_is_accept], y[~_is_accept]

def gen_MH_samples(target_density, porposal_density, proposal_rand, x0, n=1000):
    prev_x = x0
    while n>0:
        u = random.random()
        x = proposal_rand(prev_x)
        acce_p = min(1, target_density(x)/target_density(prev_x) * 
            porposal_density(prev_x, x)/porposal_density(x, prev_x))
        if u<=acce_p:
            prev_x = x
        
        yield prev_x
        n -= 1
    
def get_ranwalk_MH_samples(target_density, x0, n=1000):
    prev_x = x0
    while n>0:
        x = random.gauss(prev_x, 1)  # Do a random walk
        acce_p = target_density(x)/target_density(prev_x)
        if acce_p >=1 or random.random()<acce_p:
            prev_x = x               # To accept new val or not
            _has_updated=1
        else:
            _has_updated=0
        
        yield prev_x, _has_updated
        n-=1

def gen_gibbs_sample(condition_density_ls, x0, n=15):
    k = len(condition_density_ls)
    assert k==len(x0)
    x = x0
    while n>0:
        for i, den_func in enumerate(condition_density_ls):
            x[i] = den_func(*x[:i-1], *x[i+1:])
        yield x
        n-=1

def gen_bivariate_sample(cond_xvers_y, cond_yvers_x, x0, y0):
    x, y = x0, y0
    while 1:
        x = cond_xvers_y(y)
        y = cond_yvers_x(x)
        _v0 = yield x, y      # Syntax: sampler.send(15)
        if _v0 is not None:
            x, y = _v0


class UniSliceSampler:
    def __init__(self, f, x0) -> None:
        '''
        Parameters:
        f -- The function proportional to the density,
            or the frenquency array sits on each samples
        x0 -- current sample point(univariate)
        u -- the vertical level defining the slice, while
            u \mid x \sim U(0, f(x))
        m -- typical estimate of a slice
        k -- 
        '''
        self.density=f
        self.x = x0
        self._intv = 'stepping'

    def use_stepping(self, m, k):
        self._intv = 'stepping'
        self.m = m
        self.k = k
    
    def _stepping_interval(self, u):
        '''
        Parameters:
        f -- The function proportional to the density,
            or the frenquency array sits on each samples
        x0 -- current sample point(univariate)
        u -- the vertical level defining the slice, while
            u \mid x \sim U(0, f(x))
        m -- typical estimate of a slice
        k -- maximum slice counts
        '''
        
        l = self.x-self.m*random.random()
        r = l+self.m

        l_nlim = self.k*random.random()//1
        r_nlim = self.k-l_nlim-1
        while l_nlim>0:
            if self.density(l)<=u:
                break
            l -= self.m
            l_nlim -= 1
        
        while r_nlim>0:
            if self.density(r)<=u:
                break
            r += self.m
            r_nlim -= 1

        return l, r

    def _shrinkage_sample(self, u, l, r):

        while True:
            x = l + random.random()*(r-l)
            if self.density(x)>u:
                return x
            if x<self.x:
                l=x
            elif x>self.x:
                r=x
            else:
                raise StopIteration
    
    def __iter__(self):
        u = self.density(self.x)*random.random()

        l, r = self._stepping_interval(u)
        x = self._shrinkage_sample(u, l, r)
        yield x
        self.x = x
