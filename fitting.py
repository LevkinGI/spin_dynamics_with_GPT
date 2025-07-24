# fitting.py
import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def fit_function_theta(t, A1_theta, f1_theta, n1_theta, A2_theta, f2_theta, n2_theta, f1_GHz, f2_GHz):
    w1 = 2 * np.pi * f1_GHz * 1e9
    w2 = 2 * np.pi * f2_GHz * 1e9
    res = np.empty_like(t)
    for i in range(t.shape[0]):
        res[i] = (A1_theta * np.sin(w1 * t[i] + f1_theta) *
                  np.exp(-w1 * t[i] / (2 * np.pi) / n1_theta) +
                  A2_theta * np.sin(w2 * t[i] + f2_theta) *
                  np.exp(-w2 * t[i] / (2 * np.pi) / n2_theta))
    return res

@njit(cache=True, fastmath=True)
def fit_function_phi(t, A1_phi, f1_phi, n1_phi, A2_phi, f2_phi, n2_phi, f1_GHz, f2_GHz):
    w1 = 2 * np.pi * f1_GHz * 1e9
    w2 = 2 * np.pi * f2_GHz * 1e9
    res = np.empty_like(t)
    for i in range(t.shape[0]):
        res[i] = (A1_phi * np.sin(w1 * t[i] + f1_phi) *
                  np.exp(-w1 * t[i] / (2 * np.pi) / n1_phi) +
                  A2_phi * np.sin(w2 * t[i] + f2_phi) *
                  np.exp(-w2 * t[i] / (2 * np.pi) / n2_phi))
    return res

def residuals_stage1(params, t, theta_data, phi_data, A1_theta, A2_theta, A1_phi, A2_phi):
    f1_theta, n1_theta, f2_theta, n2_theta, f1_phi, n1_phi, f2_phi, n2_phi, f1_GHz, f2_GHz = params
    theta_model = fit_function_theta(t, A1_theta, f1_theta, n1_theta,
                                     A2_theta, f2_theta, n2_theta, f1_GHz, f2_GHz)
    phi_model = fit_function_phi(t, A1_phi, f1_phi, n1_phi,
                                 A2_phi, f2_phi, n2_phi, f1_GHz, f2_GHz)
    return np.concatenate((theta_model - theta_data, phi_model - phi_data))

def residuals_stage2(params, t, theta_data, phi_data, f1_theta, n1_theta, f2_theta, n2_theta,
                     f1_phi, n1_phi, f2_phi, n2_phi, f1_GHz, f2_GHz):
    A1_theta, A2_theta, A1_phi, A2_phi = params
    theta_model = fit_function_theta(t, A1_theta, f1_theta, n1_theta,
                                     A2_theta, f2_theta, n2_theta, f1_GHz, f2_GHz)
    phi_model = fit_function_phi(t, A1_phi, f1_phi, n1_phi,
                                 A2_phi, f2_phi, n2_phi, f1_GHz, f2_GHz)
    return np.concatenate((theta_model - theta_data, phi_model - phi_data))

def combined_residuals(params, t, theta_data, phi_data):
    (A1_theta, f1_theta, n1_theta, A2_theta, f2_theta, n2_theta,
     A1_phi, f1_phi, n1_phi, A2_phi, f2_phi, n2_phi,
     f1_GHz, f2_GHz) = params
    theta_model = fit_function_theta(t, A1_theta, f1_theta, n1_theta,
                                     A2_theta, f2_theta, n2_theta, f1_GHz, f2_GHz)
    phi_model = fit_function_phi(t, A1_phi, f1_phi, n1_phi,
                                 A2_phi, f2_phi, n2_phi, f1_GHz, f2_GHz)
    return np.concatenate((theta_model - theta_data, phi_model - phi_data))
