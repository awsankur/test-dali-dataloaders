import numpy as np

def check_nan(x):
    if np.isnan(x).any():
        return True
    else:
        return False

def check_zero(x):
    x_0 = np.where(x == 0)
    if len(x_0[0]) > 0:
        print("x_0: {}".format(x_0))
        return True
    

def normalize_numpy_0_to_1(x):
    x_min = x.min(axis=(0,1), keepdims=True)
    x_max = x.max(axis=(0,1), keepdims=True)
    diff_min_max = x_max - x_min
    if check_nan(diff_min_max):
        print("diff_min_max is nan")
    if check_nan(x-x_min):
        print("x-x_min is nan:")
    if check_nan(x):
        print("x contains nan before normalization")
    if check_zero(diff_min_max):
        print("diff_min_max has zero")
        print("x_max",x_max)
        print("x_min",x_min)
        print("diff_min_max",diff_min_max)
        print("x",x.shape)
    x = (x - x_min)/(x_max-x_min)
    if check_nan(x):
        print("x contains nan after normalization")
    return x
