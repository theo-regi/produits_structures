import numpy as np

#-------------------------------------------------------------------------------------------------------
#----------------------------Script pour implémenter les classes de produits----------------------------
#-------------------------------------------------------------------------------------------------------
#Zero-Coupon Class:: nominal (optionnal = 100 if not given)
class ZCBond:
    """
    Classe qui cherche à répliquer un bond zero coupon = 1 paiement unique à maturité.
    
    Input: Nominal (optionnal)

    - NPV par un discount factor:: discount factor
    - NPV par un taux et une maturité:: rate, maturity
    - Récupérer le taux correspondant:: discount factor, maturity

    Nous implémentons que les fonctions/données de bases, on pourrait imaginer le risque de liquidité/crédit
    """
    def __init__(self, nominal: float=100) -> None:
        self.__nominal = nominal
        pass

    def get_npv_zc_from_df(self, discount_factor: float) -> float:
        """
        Input: discount factor (ex: 0.98) as float
        
        Returns: Net Present Value of the Zero Coupon Bond.
        """
        return self.__nominal * discount_factor

    def get_npv_zc_from_zcrate(self, rate: float, maturity: float) -> float:
        """
        Calculate Net Present Value from Zero Coupon Rate
        
        Input: Zero-coupon rate, maturity of the Zero Coupon bond
        """
        return  np.exp(-rate * maturity) * self.__nominal

    def get_discount_factor_from_zcrate(self, rate: float, maturity: float) -> float:
        """
        Calculate the discount factor from the zero-coupon rate and maturity.
        
        Input: Zero-counpon rate, maturity
        """
        return np.exp(-rate * maturity)

    def get_zc_rate(self, discount_factor: float, maturity: float) -> float:
        """
        Returns Zero-Coupon rate based on the maturity and discount factor.
        
        Input: Discount factor, maturity
        """
        return -np.log(discount_factor)/maturity 

    def get_ytm(self, market_price: float, maturity: float) -> float:
        """
        Returns actual yield to maturity from market price and maturity.
        
        Input: price, maturity.
        """
        if maturity == 0:
            raise ValueError("Maturity cannot be zero when computing YTM.")
        return (self.__nominal/market_price)**(1/maturity) - 1

    def get_duration_macaulay(self, maturity: float) -> float:
        """
        Returns Macaulay duration (=maturity for a ZC)
        
        Input: maturity
        """
        return maturity

    def get_modified_duration(self, market_price: float, maturity: float) -> float:
        """
        Returns Modified Duration.
        
        Input: market price, maturity
        """
        return self.get_duration_macaulay(maturity)/(1+self.get_ytm(market_price, maturity))

    def get_sensitivity(self, new_rate: float, maturity: float) -> float:
        """
        Returns sensitivity from a zero coupon bond
        
        Input: new_rate, maturity
        """
        return self.get_duration_macaulay(maturity)/(1+new_rate)

    def get_convexity(self, maturity: float, market_price: float = None, discount_factor: float=None) -> float:
        """
        Computes the convexity of a zero-coupon bond. Works for issued ones, and non issued ones.
        
        Input:
        - T (float): Time to maturity (in years)
        - market_price (float, optional): Market price of the bond. If None, we condisder the case non issued ZC (NEED DF).
        - discount factor (float, optional): If None, we consider an issued ZC (NEED market price)
        
        Returns: float: Convexity of the zero-coupon bond.
        """

        if market_price is not None and discount_factor is None:
            ytm = self.get_ytm(market_price, maturity)
            return (maturity*(maturity+1)*self.__nominal)/(market_price*((1+ytm)**(2+maturity)))
        elif discount_factor is not None and market_price is None:
            market_price = self.get_npv_zc_from_df(discount_factor)
            ytm = self.get_ytm(market_price, maturity)
            return (maturity*(maturity+1)*self.__nominal)/(market_price*((1+ytm)**(2+maturity)))
        else:
            raise ValueError(f"Incorrect input, we need discount factor OR market price: DF = {discount_factor} and market price = {market_price}.")
      