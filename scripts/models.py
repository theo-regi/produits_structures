from products import EQDProduct
import numpy as np
from scipy.stats import norm
from constants import OptionType

#-------------------------------------------------------------------------------------------------------
#----------------------------Script pour implémenter les différents models diffusion/pricing------------
#-------------------------------------------------------------------------------------------------------
#Black-Scholes-Merton model:
class BSM:
    """
    Black-Scholes-Merton model for pricing options.
    """
    def __init__(self, option:EQDProduct, sigma:float=None) -> None:
        self._sigma = sigma
        self._option = option

    def d1(self, spot:float) -> float:
        return (np.log(spot/self._option._strike) + (self._option._rate - self._option._div_rate + (self._sigma**2)/2) * self._option.T) / (self._sigma * np.sqrt(self._option.T))

    def d2(self, spot:float) -> float:
        return self.d1(spot) - self._sigma * np.sqrt(self._option.T)

    def price(self, spot:float) -> float:
        """
        Calculate the price of the given option.
        """
        if self._option._type == OptionType.CALL:
            return spot * norm.cdf(self.d1(spot)) * np.exp(-self._option._div_rate*self._option.T) - self._option._strike * np.exp(-self._option._rate * self._option.T) * norm.cdf(self.d2(spot))
        elif self._option._type == OptionType.PUT:
            return self._option._strike * np.exp(-self._option._rate * self._option.T) * norm.cdf(-self.d2(spot)) - spot * norm.cdf(-self.d1(spot)) * np.exp(-self._option._div_rate*self._option.T)
        else:
            ValueError("Option type not supported !")
            pass

    def delta(self, spot:float) -> float:
        """
        Calculate Delta of the given option.
        """
        if self._option._type == OptionType.CALL:
            return norm.cdf(self.d1(spot)) * np.exp(-self._option._div_rate * self._option.T)
        elif self._option._type == OptionType.PUT:
            return (norm.cdf(self.d1(spot))-1) * np.exp(-self._option._div_rate * self._option.T)
        else:
            ValueError("Option type not supported !")
            pass

    def gamma(self, spot:float) -> float:
        """
        Calculate Gamma of the given option.
        """
        d1_prime = 1/np.sqrt(2*np.pi) * np.exp(-(self.d1(spot)**2)/2)
        return d1_prime * np.exp(-self._option._div_rate * self._option.T) / (spot * self._sigma * np.sqrt(self._option.T))

    def vega(self, spot:float) -> float:
        """
        Calculate Vega of the given option.
        """
        d1_prime = 1/np.sqrt(2*np.pi) * np.exp(-self.d1(spot)**2/2)
        return spot * np.sqrt(self._option.T) * d1_prime * np.exp(-self._option._div_rate * self._option.T)

    def theta(self, spot:float, rate:float=None) -> float:
        """
        Calculate Theta of the given option, in case of using the option rate for the dividend, give a risk free rate.
        """
        if rate is None:
            rate = self._option._rate
        q = rate - self._option._div_rate

        d1_prime = 1/np.sqrt(2*np.pi) * np.exp(-self.d1(spot)**2/2)

        if self._option._type == OptionType.CALL:
            return -(spot * np.exp(-self._option._div_rate * self._option.T) * d1_prime * self._sigma) / (2 * np.sqrt(self._option.T)) + q*spot*norm.cdf(self.d1(spot))* np.exp(-self._option._div_rate * self._option.T) - rate*self._option._strike*np.exp(-rate*self._option.T)*norm.cdf(self.d2(spot))
        elif self._option._type == OptionType.PUT:
            return -(spot * np.exp(-self._option._div_rate * self._option.T) * d1_prime * self._sigma) / (2 * np.sqrt(self._option.T)) - q*spot*norm.cdf(self.d1(spot))* np.exp(-self._option._div_rate * self._option.T) + rate*self._option._strike*np.exp(-rate*self._option.T)*norm.cdf(-self.d2(spot))
        else:
            ValueError("Option type not supported !")
            pass
        
    def rho(self, spot:float, rate:float=None) -> float:
        """
        Calculate Rho of the given option, in case of using the option rate for the dividend, give a risk free rate.
        """
        if rate is None:
            rate = self._option._rate
        if self._option._type == OptionType.CALL:
            return self._option._strike*self._option.T*np.exp(-rate*self._option.T)*norm.cdf(self.d2(spot))
        elif self._option._type == OptionType.PUT:
            return -self._option._strike*self._option.T*np.exp(-rate*self._option.T)*norm.cdf(-self.d2(spot))
        else:
            ValueError("Option type not supported !")
            pass


#Black 76:

#Heston model:
