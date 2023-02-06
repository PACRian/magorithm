from math import floor

import numpy as np
from func_util import POINT_INITER, versa
from matplotlib import pyplot as plt
from numpy import random as npr
from scipy.interpolate import RectBivariateSpline


def get_bounded(interval, mlim=np.inf, slim=0):
    return np.maximum(slim, interval[0]), np.minimum(mlim, interval[1])



class MapSlice:
    def __init__(self, marray=None, x0=None, 
        w_size=None, 
        ubound=None, 
        mm_func=None,
        do_subpixel=False, 
        interpolate_deg=1, 
        is_fixed_bound=True,
        is_hold_trajection=False) -> None:
        '''
        Initialization of a 2D slice sampler, with an existing map array bounded.
        Parameters:
        marray: The map array bounded to this slicer, as the 
            unnormalized probability mass function in that 2-D plane.
        x0: The initial sample point, default set to the center of whole array,
            also can be set to a callable object `init_point_getter(arr)`
            which takes `self.marray` as the input.
        w_size: The initial interval size for shrinkage searching, 
            can be a number or a binary tuple `(h_size, w_size)`, 
            default set to `None` means that bounds is fixed 
            as the complete array index range.
        ubound: The limit for the joint proposal random variable u, where
            $$ u\mid x \sim Unif(0, f(x)) $$
        mm_func: the probability mass function of a given map(`marray` or self determined)
        _do_semipixel: Enable sub-pixel tracking or just pixel level
        
        '''
        self.marray = marray
        self.init_p = x0

        self.interval = w_size
        self.ubound = ubound

        self.mapfunc = mm_func if callable(mm_func) else \
            self._mapfunc(self.marray, do_subpixel, interpolate_deg)
        
        self._isfixed = is_fixed_bound
        self._isheld = is_hold_trajection

        self._counter = 0
        self._upcounts= None
    

    def sample(self, n=10):
        self._upcounts = n
        return np.array([p for p in self])

    @staticmethod
    def _mapfunc(arr, do_subpixel, deg):
        assert isinstance(deg, int) and deg>0
        
        if bool(do_subpixel):
            h, w = arr.shape
            _getter = RectBivariateSpline(np.arange(h), np.arange(w), arr, 
                kx=deg, ky=deg)
        else:
            _getter = lambda i,j: arr[floor(i), floor(j)]

        return lambda i, j: _getter(i, j) 
        # CAUTION: 
        # `ij` notation used, if try to access an array value lies on 
        # (x, y) in the 2-dimensional coordination, use below:
        # > sampler = MapSlice()
        # > mapfunc = sampler.mapfunc
        # > x, y = 1.1, 2.3
        # > v = mapfunc(y, x)
    
    def _deal_end(self):
        self._counter=0
        self._upcounts=None

    def __iter__(self):
        x, y = self.init_p
        while 1:
            # Get u
            u = self._uval_rand([x, y])

            # Update x
            x = self._shrink_rand(u, [x, y], fixed='y')

            # Update y
            y = self._shrink_rand(u, [x, y], fixed='x')

            yield x, y, u

            self._counter+=1
            if self._upcounts is not None and self._counter==self._upcounts:
                self._deal_end()
                return  
        
    @property
    def marray(self):
        return self._marray

    @property
    def interval(self):
        if self._isfixed:
            lrs = ([0, 0], self._wsize)
        else: 
            lrs = get_bounded(self._int_rand(), mlim=self.marray.shape)

        return np.vstack(lrs)
    
    @property
    def init_p(self):
        return self._init_point

    @marray.setter
    def marray(self, marray):
        if marray is None:
            return 

        if not isinstance(marray, np.ndarray):
            raise ValueError("`marray` must be a numpy array")

        if marray.ndim != 2:
            raise ValueError("`marray` must be a 2 dimensional array")
        self._marray = marray
    
    @interval.setter
    def interval(self, w_size):
        if w_size is None:
            w_size = self.marray.shape
        elif len(w_size)==1 and isinstance(w_size, (int, float)):
            w_size = (w_size, w_size)
        elif len(w_size)> 2:
            raise ValueError("w_size should be a scalar or 2-D array")
        
        self._wsize = np.array(w_size)
    
    @init_p.setter
    def init_p(self, p):
        # Two choice, array like or callable
        if p is None:
            p = 'abovemean'

        if isinstance(p, (list, tuple, np.ndarray)):
            p = np.array(p).ravel()
            assert p.size==2, "The format of initial point should be `(x0, y0)`"
            self._init_point = p
            return
        elif isinstance(p, str):
            assert p in ('top', 'randpicker', 'midpoint', 'abovemean'), "Not valid input string"
            p = lambda arr: versa(POINT_INITER[p](arr))
        elif callable(p):
            pass
        else: 
            raise ValueError("The initial point must be a callable object or two-dimensional array")
        
        self._init_point = p(self.marray)

    def _int_rand(self):
        x = self.cpt
        bias = self._wsize * npr.rand(2)
        return np.vstack((x-bias, x+self._wsize-bias))
    
    def _uval_rand(self, p0):
        u = npr.rand()*self.mapfunc(p0[1], p0[0])

        if self.ubound is not None:
            u = min(max(u, self.ubound[0]), self.ubound[1])
        return u

    def _shrink_rand(self, u, p0, fixed='x'):
        '''
        p0 is the format of `(x, y)`
        Note that x-axis corresponds to the j-dimension(Second array index)
        while the y-axis to the i-dimension
        '''

        if fixed == 'x':
            v0 = p0[1]
            _mm_getter = lambda y: self.mapfunc(y, p0[0])
            l, r = self.interval[:, 0]
        elif fixed == 'y':
            v0 = p0[0]
            _mm_getter = lambda x: self.mapfunc(p0[1], x)
            l, r = self.interval[:, 1]
            
        while True:
            new_v = l + (r-l)*npr.rand()
            if _mm_getter(new_v)>=u:
                return new_v
            
            if new_v > v0:
                r = new_v
            elif new_v < v0:
                l = new_v
            


        


        

