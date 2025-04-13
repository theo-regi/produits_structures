import numpy as np
from scipy.stats import norm
from constants import OptionType, NUMBER_PATHS_H, NB_STEPS_H

#-------------------------------------------------------------------------------------------------------
#----------------------------Script pour implémenter les différents models diffusion/pricing------------
#-------------------------------------------------------------------------------------------------------
#Black-Scholes-Merton model:
class BSM:
    """
    Black-Scholes-Merton model for pricing options.
    """
    def __init__(self, option, sigma:float=None) -> None:
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
class Heston:
    """
    Heston Model: for options pricing, preferably autocalls, vanilla call/puts (avoid for short term maturities).
    /!\ Recommanded for derivatives when using stocastic volatilities,
    /!\ Not recommanded for digits / barriers options.

    Input:
    - 
    """
    def __init__(self, option, params:dict, nb_paths:float=NUMBER_PATHS_H, nb_steps:float=NB_STEPS_H) -> None:
        self._nb_paths=nb_paths
        self._nb_steps=nb_steps
        self._option=option
        self._v0,self._kappa,self._theta,self._eta,self._rho,=list(params.values())

        self._payoffs=[]
        self._spots=[]
        pass

    def price(self,spot:float):
        """
        Calculate the price of the given option.
        """
        self.calculate_pay_offs(spot)
        return np.exp(-self._option._rate*self._option.T) * np.mean(self._payoffs)

    def delta(self, spot:float) -> float:
        """
        Calculate Delta of the given option.
        """
        pass

    def gamma(self, spot:float) -> float:
        """
        Calculate Gamma of the given option.
        """
        pass

    def vega(self, spot:float) -> float:
        """
        Calculate Vega of the given option.
        """
        pass

    def theta(self, spot:float, rate:float=None) -> float:
        """
        Calculate Theta of the given option, in case of using the option rate for the dividend, give a risk free rate.
        """
        pass
        
    def rho(self, spot:float, rate:float=None) -> float:
        """
        Calculate Rho of the given option, in case of using the option rate for the dividend, give a risk free rate.
        """
        pass

    def calculate_pay_offs(self, spot:float) -> list:
        """
        Computes the pay_off of the given option.

        Input: Spot price of the underlying asset.
        """
        paths=self._generate_paths(spot)
        self._spots=paths['Spots'][:,-1]

        if self._option._type == OptionType.CALL:
            self._payoffs=np.maximum(self._spots-self._option._strike, 0.0)
            pass
        elif self._option._type == OptionType.PUT:
            self._payoffs=np.maximum(self._option._strike-self._spots, 0.0)
            pass
        else:
            ValueError("Option type not supported !")
            pass

    def _generate_paths(self, spot:float) -> np.ndarray:
        """
        Generate paths for the Heston model using the AES method.
        """
        def CIR_Sample(nb_paths,kappa,eta,theta,s,t,v_s):
            delta = 4.0 *kappa*theta/eta/eta
            c= 1.0/(4.0*kappa)*eta*eta*(1.0-np.exp(-kappa*(t-s)))
            kappaBar = 4.0*kappa*v_s*np.exp(-kappa*(t-s))/(eta*eta*(1.0-np.exp(-kappa*(t-s))))
            sample = c* np.random.noncentral_chisquare(delta,kappaBar,nb_paths)
            return  sample

        z1 = np.random.normal(0,1,size=(self._nb_paths, self._nb_steps))
        w1 = np.zeros((self._nb_paths, self._nb_steps+1))
        v = np.zeros((self._nb_paths, self._nb_steps+1))
        s = np.zeros((self._nb_paths, self._nb_steps+1))
        v[:,0] = self._v0
        s[:,0] = np.log(spot)

        t=np.zeros((self._nb_steps+1))
        dt = self._option.T/self._nb_steps

        for i in range(0, self._nb_steps):
            if self._nb_paths > 1:
                z1[:,i] * (z1[:,i] - np.mean(z1[:,i]))/np.std(z1[:,i])
            w1[:,i+1]=w1[:,i]+np.power(dt,0.5)*z1[:,i]

            v[:,i+1]=CIR_Sample(self._nb_paths, self._kappa, self._eta, self._theta, 0, dt, v[:,i])
            k0 = (self._option._rate - self._rho/self._eta*self._kappa*self._theta)*dt
            k1 = (self._rho*self._kappa/self._eta-0.5)*dt - self._rho/self._eta
            k2 = self._rho/self._eta
            s[:,i+1]=s[:,i]+k0 + k1*v[:,i] + k2*v[:,i+1] + np.sqrt((1-self._rho**2)*v[:,i])*(w1[:,i+1]-w1[:,i])
            t[i+1]=t[i]+dt

        sts = np.exp(s)
        paths = {"time":t, "Spots": sts}
        return paths
