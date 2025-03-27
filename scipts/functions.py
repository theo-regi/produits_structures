from constants import SOLVER_METHOD
import pandas as pd
import numpy as np
from scipy.optimize import minimize

def nelson_siegel(t, beta0, beta1, beta2, lam):
    """Nelson-Siegel model function."""
    return beta0 + beta1 * (1 - (np.exp(-t*lam) / (lam*t))) + beta2 * (((1 - np.exp(-lam*t)) / (lam*t)) - np.exp(-t / lam))

def objective_function(params, t, y):
    """Objective function to minimize."""
    beta0, beta1, beta2, lam = params
    y_hat = nelson_siegel(t, beta0, beta1, beta2, lam)
    return np.sum((y - y_hat) ** 2)

def optimize_nelson_siegel(maturities, rates):
    """Optimize Nelson-Siegel parameters."""
    initial_params = [1, 1, 1, 1]  # Initial guess for beta0, beta1, beta2, tau
    result = minimize(objective_function, initial_params, args=(maturities, rates), method=SOLVER_METHOD)
    return result.x