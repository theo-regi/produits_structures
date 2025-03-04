import numpy as np
from abc import ABC, abstractmethod
from utils import Maturity_handler, PaymentScheduleHandler, Rates_curve
import utils

#-------------------------------------------------------------------------------------------------------
#----------------------------Script pour implémenter les classes de produits----------------------------
#-------------------------------------------------------------------------------------------------------

#____________________________Classe pour les ZC (pas de call de l'abstraite)_____________________________
#Zero-Coupon Class:: nominal (optionnal = 100 if not given)
class ZCBond():
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
      
#-------------------------------------------------------------------------------------------------------
#----------------------------Classes de produits généralistes en fixed-income---------------------------
#-------------------------------------------------------------------------------------------------------
#____________________________Classe abstraite pour les produits d'Income________________________________
class FixedIncomeProduct(ABC):
    """
    Abstract class for fixed-income products:

    Input:
    - forward rates curve (dict, non optionnal)
    - start date (string, non optionnal)
    - end date (string, non optionnal)
    - paiments frequency (string, non optionnal)
    - day count convention (string, optionnal, equal to 30/360 if not provided)
    - rolling convention (string, optionnal, equal to Modified Following if not provided)
    - discounting curve to discount with a different curve than the forward rates curve (dict, optionnal)
    - notional (float, optionnal, will quote in percent if not provided)

    Abstract class to build the different types of legs for fixed income instruments.
    For fixed income leg, rate_curve will be a flat rate curve.
    """
    def __init__(self, rate_curve:dict, start_date:str, end_date:str,
                 paiement_freq:str, currency:str, day_count:str=30/360, rolling_conv:str="Modified Following",
                 discounting_curve:dict=None, notional:float=100) -> None:
        
        self.__rate_curve=rate_curve
        self.__start_date=start_date
        self.__end_date=end_date
        self.__paiement_freq=paiement_freq
        self.__currency=currency
        self.__day_count=day_count
        self.__rolling_conv=rolling_conv
        if discounting_curve is None:
            self.__discounting_curve=rate_curve
        else:
            self.__discounting_curve=discounting_curve
        self.__notional = notional
    
        self.__cashflows = {}
        self.__paiments_schedule = \
            PaymentScheduleHandler(self.__start_date, self.__end_date,
            self.__paiement_freq, self.__day_count,self.__rolling_conv).build_schedule(\
            convention=self.__day_count, rolling_convention=self.__rolling_conv, market=\
            utils.get_market(currency=self.__currency))
        

    @abstractmethod
    def calculate_npv(self) -> float:
        """
        Returns the product NPV as float
        """
        return sum(entry["NPV"] for entry in self.__cashflows.values())

    @abstractmethod
    def calculate_duration(self) -> float:
        """
        Returns duration of the product
        """
        duration_ti = sum(entry["NPV"] * entry["Year_Fraction"] for entry in self.__cashflows.values())
        return duration_ti / self.calculate_npv()

    @abstractmethod
    def calculate_sensitivity(self) -> float:
        """
        Returns sensitivity of the product
        """
        pass

    @abstractmethod
    def calculate_convexity(self) -> float:
        """
        Returns convexity of the product
        """
        pass

    @abstractmethod
    def calculate_pv01(self) -> float:
        """
        Returns pov01 of the product
        """
        return sum(entry["PV01"] for entry in self.__cashflows.values())
  
class FixedLeg(FixedIncomeProduct):
    """
    Class pour une leg fixe, on va pouvoir calculer le npv, duration, convexity, pv01, etc.
    Utilisée pour les swaps et fixed bonds.

    Input:
    - forward rates curve (dict, non optionnal)
    - start date (string, non optionnal)
    - end date (string, non optionnal)
    - paiments frequency (string, non optionnal)
    - day count convention (string, optionnal, equal to 30/360 if not provided)
    - rolling convention (string, optionnal, equal to Modified Following if not provided)
    - discounting curve to discount with a different curve than the forward rates curve (dict, optionnal)
    - notional (float, optionnal, will quote in percent if not provided)

    Returns: class with functions of NPV, duration, convexity, pv01, etc.
    For fixed income leg, rate_curve will be a flat rate curve.
    For bonds, notional exchange will be True in build_cashflows.
    """
    def __init__(self, rate_curve:dict, start_date:str, end_date:str,
                 paiement_freq:str, day_count:str=30/360, rolling_conv:str="Modified Following",
                 discounting_curve:dict=None, notional:float=100) -> None:
        super().__init__(rate_curve, start_date, end_date, paiement_freq, day_count, rolling_conv, discounting_curve, notional)
        pass

    def calculate_npv(self) -> float:
        """
        Calculate the NPV of the fixed leg.
        """
        return super().calculate_npv()

    def calculate_duration(self) -> float:
        """
        Calculate the duration of the fixed leg.
        """
        return super().calculate_duration()

    def calculate_sensitivity(self) -> float:
        """
        Calculate the sensitivity of the fixed leg.
        """
        new_rate = self.__rate_curve[self.__end_date] + 0.01
        return self.calculate_duration() / (1 + new_rate)

    def calculate_convexity(self) -> float:
        """
        Calculate the convexity of the fixed leg.
        """
        pass

    def calculate_pv01(self) -> float:
        """
        Calculate the PV01 of the fixed leg.
        """
        return super().calculate_pv01()
    
    def build_cashflows(self, exchange_notionnal:str=False) -> dict:
        """
        Build the paiements schedule for the fixed leg.
        Input:
        - exchange_notionnal (string, optionnal, equal to False if not provided), provide True for bonds.
        """
        for date in self.__paiments_schedule:
            if date != self.__end_date:
                npv = self.__notional * self.__rate_curve[date] * self.__paiments_schedule[date]["Year_Fraction"] * self.__discounting_curve[date]
                pv01 = npv * 1/10000
                self.__cashflows[date] = {"NPV": npv, "PV01": pv01}
            else:
                if exchange_notionnal:
                    npv = self.__notional * self.__rate_curve[date] * self.__paiments_schedule[date]["Year_Fraction"] * self.__discounting_curve[date] + self.__notional * self.__discounting_curve[date]
                    pv01 = npv * 1/10000
                    self.__cashflows[date] = {"NPV": npv, "PV01": pv01}
                else:
                    npv = self.__notional * self.__rate_curve[date] * self.__paiments_schedule[date]["Year_Fraction"] * self.__discounting_curve[date]
                    pv01 = npv * 1/10000
                    self.__cashflows[date] = {"NPV": npv, "PV01": pv01}
        pass

    def calculate_yield(self) -> float:
        """
        Calculate the yield of the fixed leg.
        """
        pass

class FloatLeg(FixedIncomeProduct):
    """
    Class pour une leg flottante, on va pouvoir calculer le npv, duration, convexity, pv01, etc.
    Utilisée pour les swaps et FRNs.

    Input:
    - forward rates curve (dict, non optionnal)
    - start date (string, non optionnal)
    - end date (string, non optionnal)
    - paiments frequency (string, non optionnal)
    - day count convention (string, optionnal, equal to 30/360 if not provided)
    - rolling convention (string, optionnal, equal to Modified Following if not provided)
    - discounting curve to discount with a different curve than the forward rates curve (dict, optionnal)
    - notional (float, optionnal, will quote in percent if not provided)
    """
    def __init__(self, rate_curve:dict, start_date:str, end_date:str,
                 paiement_freq:str, day_count:str=30/360, rolling_conv:str="Modified Following",
                 discounting_curve:dict=None, notional:float=100) -> None:
        super().__init__(rate_curve, start_date, end_date, paiement_freq, day_count, rolling_conv, discounting_curve, notional)
        pass
    
    def calculate_npv(self) -> float:
        """
        Calculate the NPV of the float leg.
        """
        return super().calculate_npv()
    
    def calculate_duration(self) -> float:
        """
        Calculate the duration of the float leg.
        """
        return super().calculate_duration()
    
    def calculate_sensitivity(self) -> float:
        """
        Calculate the sensitivity of the float leg.
        """
        pass

    def calculate_convexity(self) -> float:
        """
        Calculate the convexity of the float leg.
        """
        pass

    def calculate_pv01(self) -> float:
        """
        Calculate the PV01 of the float leg.
        """
        return super().calculate_pv01()

"""
1: On va utiliser cette classe abstraite pour tout les produits composés de ZC:
- Float et Fix leg, car ce seront des bases pour construire le rest, il faut un input échange de notionnel (yes/no)
    - Fixed bond(c'est une leg fix avec de l'échange de notionnel à la fin)
    - FRNs (pareil que fixed bond mais en float)
- Caps / floors
- Options de call / put

2: Classe portfolio fixed income qui permettrat de construire les produits ci-dessus:
- Fix to Float swap (mono CCY)
- Fix to fix swap (XCCY)
- float to float swap (XCCY)
- FRN + cap (ou / +) floor
- Callable / puttable
- pricing d'oblig au marché (égalisation d'une npv fix sur un float leg)
-> On aura besoin de mettre les legs / produits en 1: vendeur ou acheteur et de build pas mal de fonction
    de valorisation / risque
"""