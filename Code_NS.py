import numpy as np
import pandas as pd

def nelson_siegel(
    t:float, 
    beta0_coeff:float, 
    beta1_coeff:float, 
    beta2_coeff:float, 
    lambda_coeff:float
    ):
    return beta0_coeff + \
        beta1_coeff * (1 - np.exp(-t / lambda_coeff)) / (t / lambda_coeff) + \
        beta2_coeff * ((1 - np.exp(-t / lambda_coeff)) / (t / lambda_coeff) - \
                            np.exp(-t / lambda_coeff))


# Calibration
df=pd.read_csv('RateCurve.csv', sep=";")
def convert_mat(pillar):
    if "M" in pillar:
        return int(pillar.replace("M", "")) / 12
    if "Y" in pillar:
        return int(pillar.replace("Y", ""))
    raise ValueError("Unknown format")
df["maturity"]=df["Pillar"].map(convert_mat)

maturities=list(df["maturity"])
rates=list(df["Rate"])

from scipy.optimize import curve_fit
popt, _=curve_fit(nelson_siegel, maturities, rates)
print(popt)