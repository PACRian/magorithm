from math import floor

import numpy as np
from numpy import random as npr

# 初始点生成器
# 一般格式:
# initer(arr) -- arr: The map array to be sampled
_select_top = lambda _ : [0, 0]
_select_midpoint = lambda arr: [round(i/2) for i in arr.shape]
_select_ranpicker = lambda arr: [npr.randint(i) for i in arr.shape]
def _select_abovemean(arr):
    xind, yind = np.where(arr>=arr.mean())
    v = npr.randint(len(xind))
    return xind[v], yind[v]

POINT_INITER = {
    "top": _select_top,
    "midpoint": _select_midpoint, 
    "randpicker":_select_ranpicker, 
    "abovemean":_select_abovemean}

# 将(i,j)形式转为(x,y)格式
versa = lambda a: a[::-1]