import math
from typing import List

import numpy as np
from fenics import Function, Dx, project

def grad(f):
    return Dx(f, 0)

#================================================================================================
# finite differences to approximate first derivatives
#================================================================================================

def finite_backwards_difference(n: int, k: int, u: Function, u_old_arr: List[Function], dt: float):
    """
    Approximate nth derivative by finite backwards difference of order k of given fenics function z
    """
    # Weighting coefficients and quintile idx
    c_arr, s_arr = finite_difference_coefficients(n, k)

    u_t = 0
    # Add terms to finite backwards difference from most past to most recent time point
    for s, c in zip(s_arr, c_arr):
        if s == 0:
            u_t += c * u
        else:
            u_t += c * u_old_arr[s]

    u_t = u_t / dt**n

    return u_t


def finite_difference_coefficients(n: int, k: int):
    '''
    Calculates weighting coefficients for finite backwards difference of order k for nth derivative
    '''
    # Number points for required for nth derivative of kth order
    N = n + k

    # Point indexes [-k, ..., -1, 0]
    # 0 = current, -1 = previous time point, ...
    s_arr = np.arange(- N + 1, 1)
    A = np.vander(s_arr, increasing=True).T
    b = np.zeros(N)
    b[n] = math.factorial(n)

    # Weighting coefficients of quintiles
    c_arr = np.linalg.solve(A, b)
    # Fenics can't handle numpy arrays
    c_arr = c_arr.tolist()

    return c_arr, s_arr

#================================================================================================
# Decorator to assemble output
#================================================================================================
def tag_function_space(fs):
   def decorator(method):
        method.function_space = fs  # Add the tag attribute to the method
        return method
   return decorator



