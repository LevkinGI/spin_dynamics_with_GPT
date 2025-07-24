# fitting_cy.pyx
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3
import numpy as np
cimport numpy as np
from libc.math cimport sin, exp, M_PI

def fit_function_theta(np.ndarray[double, ndim=1] t, double A1_theta, double f1_theta, double n1_theta,
                       double A2_theta, double f2_theta, double n2_theta, double f1_GHz, double f2_GHz):
    cdef Py_ssize_t N = t.shape[0]
    cdef np.ndarray[double, ndim=1] res = np.empty(N, dtype=np.float64)
    cdef double w1 = 2 * M_PI * f1_GHz * 1e9
    cdef double w2 = 2 * M_PI * f2_GHz * 1e9
    cdef Py_ssize_t i
    for i in range(N):
         res[i] = A1_theta * sin(w1 * t[i] + f1_theta) * exp(-w1 * t[i] / (2 * M_PI) / n1_theta) + \
                  A2_theta * sin(w2 * t[i] + f2_theta) * exp(-w2 * t[i] / (2 * M_PI) / n2_theta)
    return res

def fit_function_phi(np.ndarray[double, ndim=1] t, double A1_phi, double f1_phi, double n1_phi,
                     double A2_phi, double f2_phi, double n2_phi, double f1_GHz, double f2_GHz):
    cdef Py_ssize_t N = t.shape[0]
    cdef np.ndarray[double, ndim=1] res = np.empty(N, dtype=np.float64)
    cdef double w1 = 2 * M_PI * f1_GHz * 1e9
    cdef double w2 = 2 * M_PI * f2_GHz * 1e9
    cdef Py_ssize_t i
    for i in range(N):
         res[i] = A1_phi * sin(w1 * t[i] + f1_phi) * exp(-w1 * t[i] / (2 * M_PI) / n1_phi) + \
                  A2_phi * sin(w2 * t[i] + f2_phi) * exp(-w2 * t[i] / (2 * M_PI) / n2_phi)
    return res
