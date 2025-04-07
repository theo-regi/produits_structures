from constants import OptionType, BASE_SPOT, BASE_STRIKE, BASE_RATE, BASE_CURRENCY, \
    BASE_DIV_RATE, BOUNDS_MONEYNESS, OTM_CALIBRATION, VOLUME_CALIBRATION, VOLUME_THRESHOLD, \
    INITIAL_SSVI, SSVI_METHOD, OPTIONS_SOLVER_SSVI,BASE_NOTIONAL, CONVENTION_DAY_COUNT, \
    ROLLING_CONVENTION, FORMAT_DATE, TYPE_INTERPOL, EXCHANGE_NOTIONAL, BASE_SHIFT,BASE_MODEL,\
    BASE_SIGMA, BASE_METHOD_VOL, TOLERANCE, MAX_ITER, BOUNDS, STARTING_POINT

from scipy.optimize import minimize
import numpy as np
from abc import ABC, abstractmethod
from utils import PaymentScheduleHandler, Rates_curve
import utils as utils
from models import BSM
from utils import ImpliedVolatilityFinder, SVIParamsFinder
import pandas as pd
from scipy.stats import norm
from collections import defaultdict

#Supported models for options pricing
dict_models = {"Black-Scholes-Merton": BSM}

#-------------------------------------------------------------------------------------------------------
#----------------------------Script pour implémenter les classes de produits----------------------------
#-------------------------------------------------------------------------------------------------------

#____________________________Classe pour les ZC (pas de call de l'abstraite)_____________________________
#Zero-Coupon Class:: nominal (optional = 100 if not given)
class ZCBond():
    """
    Classe qui cherche à répliquer un bond zero coupon = 1 paiement unique à maturité.
    
    Input: Nominal (optional)

    - NPV par un discount factor:: discount factor
    - NPV par un taux et une maturité:: rate, maturity
    - Récupérer le taux correspondant:: discount factor, maturity

    Nous implémentons que les fonctions/données de bases, on pourrait imaginer le risque de liquidité/crédit
    """
    def __init__(self, nominal: float=BASE_NOTIONAL) -> None:
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
    - forward rates curve (dict, non optional)
    - start date (string, non optional)
    - end date (string, non optional)
    - paiments frequency (string, non optional)
    - day count convention (string, optional, equal to 30/360 if not provided)
    - rolling convention (string, optional, equal to Modified Following if not provided)
    - discounting curve to discount with a different curve than the forward rates curve (dict, optional)
    - notional (float, optional, will quote in percent if not provided)

    Abstract class to build the different types of legs for fixed income instruments.
    For fixed income leg, rate_curve will be a flat rate curve.
    """
    def __init__(self, rate_curve: Rates_curve, start_date:str, end_date:str,
                 paiement_freq:str, currency:str, day_count:str=CONVENTION_DAY_COUNT, rolling_conv:str=ROLLING_CONVENTION,
                 discounting_curve:Rates_curve=None, notional:float=BASE_NOTIONAL, spread:float=0, format:str=FORMAT_DATE, interpol: str=TYPE_INTERPOL, exchange_notional: str=EXCHANGE_NOTIONAL) -> None:
        
        self._rate_curve=rate_curve
        self._start_date=start_date
        self._end_date=end_date
        self._paiement_freq=paiement_freq
        self._currency=currency
        self._day_count=day_count
        self._discounting_curve = discounting_curve
        self._rolling_conv=rolling_conv
        self._notional = notional
        self._format = format
        self._cashflows = {}
        self._cashflows_cap ={}
        self._cashflows_floor ={}
        self._cashflows_r = {}
        self._cashflows_cap_r ={}
        self._cashflows_floor_r ={}
        self._exchange_notional = exchange_notional
        self._spread = spread
        self._interpol = interpol
        self._paiments_schedule = \
            PaymentScheduleHandler(self._start_date, self._end_date,
            self._paiement_freq, self._format).build_schedule(\
            convention=self._day_count, rolling_convention=self._rolling_conv, market=\
            utils.get_market(currency=self._currency))

    @abstractmethod
    def calculate_npv(self,cashflows) -> float:
        """
        Returns the product NPV as float
        """
        return sum(entry["NPV"] for entry in cashflows.values())
    
    @abstractmethod
    def calculate_duration(self) -> float:
        """
        Returns duration of the product
        """
        duration_ti = sum(value["NPV"] * key for key, value in self._cashflows.items())
        return duration_ti / self.calculate_npv(self._cashflows)

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
        return sum(entry["PV01"] for entry in self._cashflows.values())

#Fixed Leg Income product
class FixedLeg(FixedIncomeProduct):
    """
    Class pour une leg fixe, on va pouvoir calculer le npv, duration, convexity, pv01, etc.
    Utilisée pour les swaps et fixed bonds.

    Input:
    - forward rates curve (dict, non optional)
    - start date (string, non optional)
    - end date (string, non optional)
    - paiments frequency (string, non optional)
    - day count convention (string, optional, equal to 30/360 if not provided)
    - rolling convention (string, optional, equal to Modified Following if not provided)
    - discounting curve to discount with a different curve than the forward rates curve (dict, optional)
    - notional (float, optional, will quote in percent if not provided)

    Returns: class with functions of NPV, duration, convexity, pv01, etc.
    For fixed income leg, rate_curve will be a flat rate curve.
    For bonds, notional exchange will be True in build_cashflows.
    """
    def __init__(self, rate_curve: Rates_curve, start_date:str, end_date:str, paiement_freq:str, currency:str, day_count:str=CONVENTION_DAY_COUNT, rolling_conv:str=ROLLING_CONVENTION, discounting_curve:Rates_curve=None, notional:float=BASE_NOTIONAL, shift:float=0, format:str=FORMAT_DATE, interpol: str=TYPE_INTERPOL, exchange_notional: str=EXCHANGE_NOTIONAL) -> None:
        super().__init__(rate_curve, start_date, end_date, paiement_freq, currency, day_count, rolling_conv, discounting_curve, notional, shift, format, interpol, exchange_notional)
    
        self._rates_c = self._rate_curve.create_product_rate_curve(self._paiments_schedule, "Flat")

        if discounting_curve is None:
            self._discounting_c=self._rate_curve
        else:
            self._discounting_c=discounting_curve

        self._discountings=self._discounting_c.create_product_rate_curve(self._paiments_schedule, TYPE_INTERPOL)

        self._ZC = ZCBond(self._notional)
        self._rate_dict = dict(zip(self._rates_c["Year_fraction"], self._rates_c["Rate"]))
        self._discount_dict = dict(zip(self._discountings["Year_fraction"], self._ZC.get_discount_factor_from_zcrate(self._discountings["Rate"]/100, self._discountings["Year_fraction"])))

        self.build_cashflows()
        self.build_cashflows_npv()
        pass

    def calculate_npv(self,cashflows) -> float:
        """
        Calculate the NPV of the fixed leg.
        """
        return super().calculate_npv(cashflows)

    def calculate_duration(self) -> float:
        """
        Calculate the duration of the fixed leg.
        """
        return super().calculate_duration()

    def calculate_sensitivity(self, shift:dict=None) -> float:
        """
        Calculate the sensitivity of the fixed leg.

        Input:
        - shift (dict, optional): dictionnary of shift for each date, if not given -> linear shift of 1bps.
        """
        if shift is None:
            s = np.ones(len(self._paiments_schedule)) * BASE_SHIFT
            shift = dict(zip(self._paiments_schedule, s))
        shifted_curve = self._discounting_c.deep_copy()
        shifted_curve.shift_curve(shift, self._interpol)
        shift_fixed_leg = FixedLeg(self._rate_curve, self._start_date, self._end_date, self._paiement_freq, self._currency, self._day_count, self._rolling_conv, shifted_curve, self._notional, self._spread, self._format, self._interpol, self._exchange_notional)
        shift_fixed_leg.calculate_npv(self._cashflows)
        return shift_fixed_leg.calculate_npv(shift_fixed_leg._cashflows) - self.calculate_npv(self._cashflows)

    def calculate_convexity(self, shift:dict=None) -> float:
        """
        Calculate the convexity of the fixed leg.

        Input:
        - shift (dict, optional): dictionnary of shift for each date, if not given -> linear shift of 1bps (0.01 input).
        """
        if shift is None:
            s = np.ones(len(self._paiments_schedule)) * BASE_SHIFT
            shift = dict(zip(self._paiments_schedule, s))
        
        shifted_pos_curve = self._discounting_c.deep_copy()
        shifted_pos_curve.shift_curve(shift, self._interpol)
        shift_leg_pos = FixedLeg(self._rate_curve, self._start_date, self._end_date, self._paiement_freq, self._currency, self._day_count, self._rolling_conv, shifted_pos_curve, self._notional, self._spread, self._format, self._interpol, self._exchange_notional)
        
        neg_shift = {key: -value for key, value in shift.items()}
        shifted_neg_curve = self._discounting_c.deep_copy()
        shifted_neg_curve.shift_curve(neg_shift, self._interpol)
        shift_leg_neg = FixedLeg(self._rate_curve, self._start_date, self._end_date, self._paiement_freq, self._currency, self._day_count, self._rolling_conv, shifted_neg_curve, self._notional, self._spread, self._format, self._interpol, self._exchange_notional)
        return sum((shift_leg_pos.calculate_npv(shift_leg_pos._cashflows) + shift_leg_neg.calculate_npv(shift_leg_neg._cashflows) - 2 * self.calculate_npv(self._cashflows)) /
            ((shift[t]/100 ** 2) * self.calculate_npv(self._cashflows)) for t in self._paiments_schedule)
    
    def calculate_pv01(self) -> float:
        """
        Calculate the PV01 of the fixed leg.
        """
        return super().calculate_pv01()
    
    def build_cashflows_npv(self) -> dict:
        """
        Apply discount factors to NPV and PV01 in a cashflow dictionary.
        
        :return: New dictionary with discounted "NPV" and "PV01".
        """
        discounted_cashflows = {
            t: {
                "NPV": cf["NPV"] * self._discount_dict.get(t, 1),
                "PV01": cf["PV01"] * self._discount_dict.get(t, 1)
            }
            for t, cf in self._cashflows_r.items()
        }
        self._cashflows = discounted_cashflows
        pass

    def build_cashflows(self) -> dict:
        """
        Build the paiements schedule for the fixed leg (RAW).
        Input:
        - exchange_notionnal (string, optional, equal to False if not provided), provide True for bonds.
        """
        for i in range(len(self._paiments_schedule)-1):
            date = self._paiments_schedule[i]
            if date == self._paiments_schedule[0]:
                npv = self._notional * (self._rate_dict[date]+self._spread)/100 * date
                pv01 = self._notional * 1/10000 * date
                self._cashflows_r[date] = {"NPV": npv, "PV01": pv01}
            elif date != self._paiments_schedule[-1] and date!= self._paiments_schedule[0]:
                npv = self._notional * (self._spread+self._rate_dict[date])/100 * (date-self._paiments_schedule[i-1])
                pv01 = self._notional * 1/10000 * (date-self._paiments_schedule[i-1])
                self._cashflows_r[date] = {"NPV": npv, "PV01": pv01}
            else:
                if self._exchange_notional == True:
                    npv = self._notional * (self._spread+self._rate_dict[date])/100 * (date-self._paiments_schedule[i-1]) + self._notional
                    pv01 = self._notional * 1/10000 * (date-self._paiments_schedule[i-1])
                    self._cashflows_r[date] = {"NPV": npv, "PV01": pv01}
                else:
                    npv = self._notional * (self._spread+self._rate_dict[date])/100 * (date-self._paiments_schedule[i-1])
                    pv01 = self._notional * 1/10000  * (date-self._paiments_schedule[i-1])
                    self._cashflows_r[date] = {"NPV": npv, "PV01": pv01}
        pass

    def calculate_yield(self, market_price:float) -> float:
        """
        Calculate the yield of the fixed leg.
        """
        return utils.calculate_yield(self._cashflows_r, market_price)

#Float Leg Income product
class FloatLeg(FixedIncomeProduct):
    """
    Class pour une leg flottante, on va pouvoir calculer le npv, duration, convexity, pv01, etc.
    Utilisée pour les swaps et FRNs.

    Input:
    - forward rates curve (dict, non optional)
    - start date (string, non optional)
    - end date (string, non optional)
    - paiments frequency (string, non optional)
    - day count convention (string, optional, equal to 30/360 if not provided)
    - rolling convention (string, optional, equal to Modified Following if not provided)
    - discounting curve to discount with a different curve than the forward rates curve (dict, optional)
    - notional (float, optional, will quote in percent if not provided)
    """
    def __init__(self, rate_curve: Rates_curve, start_date:str, end_date:str,
                 paiement_freq:str, currency:str, day_count:str=CONVENTION_DAY_COUNT, rolling_conv:str=ROLLING_CONVENTION,
                 discounting_curve:Rates_curve=None, notional:float=BASE_NOTIONAL, spread:float=0, format:str=FORMAT_DATE, interpol: str=TYPE_INTERPOL, exchange_notional: str=EXCHANGE_NOTIONAL) -> None:
        super().__init__(rate_curve, start_date, end_date, paiement_freq, currency, day_count, rolling_conv, discounting_curve, notional, spread, format, interpol, exchange_notional)       
        
        self._rates_c = self._rate_curve.create_product_rate_curve(self._paiments_schedule, interpol)
        if discounting_curve is None:
            self._discounting_c=self._rate_curve
        else:
            self._discounting_c=discounting_curve

        self._discountings=self._discounting_c.create_product_rate_curve(self._paiments_schedule, interpol)
        
        self._ZC = ZCBond(self._notional)
        self._rate_dict = dict(zip(self._rates_c["Year_fraction"], self._rates_c["Forward_rate"]))
        self._discount_dict = dict(zip(self._discountings["Year_fraction"], self._ZC.get_discount_factor_from_zcrate(self._discountings["Rate"]/100, self._discountings["Year_fraction"])))
        self.build_cashflows(self._rate_dict,100, self._cashflows_r)
        self.build_cashflows_npv()
        pass
    
    def calculate_npv(self,cashflows) -> float:
        """
        Calculate the NPV of the float leg.
        """
        return super().calculate_npv(cashflows)
    
    def calculate_duration(self) -> float:
        """
        Calculate the duration of the float leg.
        """
        return super().calculate_duration()
    
    def calculate_sensitivity(self, shift_fw:dict=None, shift_discounting:dict=None) -> float:
        """
        Calculate the sensitivity of the float leg.

        Input:
        - shift_fw (dict, optional): dictionnary of shift for each date, if not given -> linear shift of 1bps.
        - shift_discounting (dict, optional): dictionnary of shift for each date, if not given -> linear shift of 1bps.
        """
        if shift_fw is None:
            s = np.ones(len(self._paiments_schedule)) * BASE_SHIFT
            shift_fw = dict(zip(self._paiments_schedule, s))

        if shift_discounting is None:
            shift_discounting = shift_fw

        shifted_fw_curve = self._rate_curve.deep_copy()
        shifted_fw_curve.shift_curve(shift_fw, self._interpol)

        shifted_discounting_curve = self._discounting_c.deep_copy()
        shifted_discounting_curve.shift_curve(shift_discounting, self._interpol)

        shift_fixed_leg = FloatLeg(shifted_fw_curve, self._start_date, self._end_date, self._paiement_freq, self._currency, self._day_count, self._rolling_conv, shifted_discounting_curve, self._notional, self._spread, self._format, self._interpol, self._exchange_notional)
        shift_fixed_leg.calculate_npv(self._cashflows)
        return shift_fixed_leg.calculate_npv(shift_fixed_leg._cashflows) - self.calculate_npv(self._cashflows)

    def calculate_convexity(self, shift_fw:dict=None, shift_discounting:dict=None) -> float:
        """
        Calculate the convexity of the float leg.

        Input:
        - shift_fw (dict, optional): dictionnary of shift for each date, if not given -> linear shift of 1bps.
        - shift_discounting (dict, optional): dictionnary of shift for each date, if not given -> linear shift of 1bps.
        """
        if shift_fw is None:
            s = np.ones(len(self._paiments_schedule)) * BASE_SHIFT
            shift_fw = dict(zip(self._paiments_schedule, s))

        if shift_discounting is None:
            shift_discounting = shift_fw
        
        def get_npv(shift_fw, shift_ds):
            shifted_fw_curve = self._rate_curve.deep_copy()
            shifted_fw_curve.shift_curve(shift_fw, self._interpol)
            shifted_discounting_curve = self._discounting_c.deep_copy()
            shifted_discounting_curve.shift_curve(shift_ds, self._interpol)
            shift_leg = FloatLeg(shifted_fw_curve, self._start_date, self._end_date, self._paiement_freq, self._currency, self._day_count, self._rolling_conv, shifted_discounting_curve, self._notional, self._spread, self._format, self._interpol, self._exchange_notional)
            return shift_leg.calculate_npv(self._cashflows)

        #Initial NPV:
        npv_0 = self.calculate_npv(self._cashflows)

        #Shift in the same direction of both curves:
        npv_pp = get_npv(shift_fw, shift_discounting)
        npv_mm = get_npv({key: -value for key, value in shift_fw.items()}, {key: -value for key, value in shift_discounting.items()})
        
        #Shift in the opposite direction of both curves:
        npv_pm = get_npv(shift_fw, {key: -value for key, value in shift_discounting.items()})
        npv_mp = get_npv({key: -value for key, value in shift_fw.items()}, shift_discounting)

        #Shift for one and not the other:
        npv_p0 = get_npv(shift_fw, {t: 0 for t in shift_discounting})
        npv_0p = get_npv({t: 0 for t in shift_fw}, shift_discounting)
        npv_m0 = get_npv({key: -value for key, value in shift_fw.items()}, {t: 0 for t in shift_discounting})
        npv_0m = get_npv({t: 0 for t in shift_fw}, {key: -value for key, value in shift_discounting.items()})

        sum_npvs = npv_pp + npv_mm - npv_p0 - npv_0p + npv_pm + npv_mp + 2 * npv_0 - npv_m0 - npv_0m

        vector_f = np.array([value for value in shift_fw.values()])
        vector_d = np.array([value for value in shift_discounting.values()])
        d = np.dot(vector_f, vector_d) * npv_0

        return sum_npvs / (d *10000)

    def calculate_pv01(self) -> float:
        """
        Calculate the PV01 of the float leg.
        """
        return super().calculate_pv01()
    
    def build_cashflows(self,dict,pourcentage,dict_result) -> dict:
        """
        Build the paiements schedule for the fixed leg.
        Input:
        - exchange_notionnal (string, optional, equal to False if not provided), provide True for bonds.
        """

        for i in range(len(self._paiments_schedule)-1):
            date = self._paiments_schedule[i]
            if date == self._paiments_schedule[0]:
                npv = self._notional * (dict[date]+self._spread)/pourcentage * date
                pv01 = self._notional * 1/10000 * date
                dict_result[date] = {"NPV": npv, "PV01": pv01}
            elif date != self._paiments_schedule[-1] and date!= self._paiments_schedule[0]:
                npv = self._notional * (dict[date]+self._spread)/pourcentage * (date-self._paiments_schedule[i-1])
                pv01 = self._notional * 1/10000 * (date-self._paiments_schedule[i-1])
                dict_result[date] = {"NPV": npv, "PV01": pv01}
            else: 
                if self._exchange_notional == True:
                    npv = self._notional * (dict[date]+self._spread)/pourcentage * (date-self._paiments_schedule[i-1]) + self._notional
                    pv01 = self._notional * 1/10000 * (date-self._paiments_schedule[i-1])
                    dict_result[date] = {"NPV": npv, "PV01": pv01}
                else:
                    npv = self._notional * (dict[date]+self._spread)/pourcentage * (date-self._paiments_schedule[i-1])
                    pv01 = self._notional * 1/10000 * (date-self._paiments_schedule[i-1])
                    dict_result[date] = {"NPV": npv, "PV01": pv01}
        pass

    def build_cashflows_npv(self) -> dict:
        """
        Apply discount factors to NPV and PV01 in a cashflow dictionary.
        
        :return: New dictionary with discounted "NPV" and "PV01".
        """
        discounted_cashflows = {
            t: {
                "NPV": cf["NPV"] * self._discount_dict.get(t, 1),
                "PV01": cf["PV01"] * self._discount_dict.get(t, 1)
            }
            for t, cf in self._cashflows_r.items()
        }
        self._cashflows = discounted_cashflows
        pass

    def calculate_yield(self, market_price:float) -> float:
        """
        Calculate the yield of the float leg.
        """
        return utils.calculate_yield(self._cashflows_r, market_price)

    def cap_value(self, cap_strike:float,sigma:float) -> float:
        """
        Calculate the cap value of the float leg.

        Input:
        - cap (float): cap value
        - sigma (float): volatility of the forward rate

        """
        df_cap= pd.DataFrame()
        df_cap["Log"] = self._rates_c["Forward_rate"].apply(lambda x: np.log((x/100)/cap_strike))
        df_cap["vol"] = (0.5*sigma**2)*self._rates_c["Year_fraction"]
        df_cap["Actu"] = sigma*np.sqrt(self._rates_c["Year_fraction"])
        df_cap["d1"] = (df_cap["Log"]+df_cap["vol"])/df_cap["Actu"]
        df_cap["d2"] = df_cap["d1"]-df_cap["Actu"]
        df_cap["value"] = self._rates_c["Forward_rate"]/100*norm.cdf(df_cap["d1"])-cap_strike*norm.cdf(df_cap["d2"])   
        self._cap_rate_dict = dict(zip(self._rates_c["Year_fraction"],  df_cap["value"]))
        self.build_cashflows(self._cap_rate_dict,1, self._cashflows_cap_r)
        self.build_cashflow_cap_npv()
        pass

    def build_cashflow_cap_npv(self):
        """
        Apply discount factors to NPV and PV01 in a cashflow dictionary.
        
        :return: New dictionary with discounted "NPV" and "PV01".
        """  
        discounted_cashflows = {
            t: {
                "NPV": cf["NPV"] * self._discount_dict.get(t, 1),
                "PV01": cf["PV01"] * self._discount_dict.get(t, 1)
            }
            for t, cf in self._cashflows_cap_r.items()
        }
        self._cashflows_cap = discounted_cashflows
        pass

    def floor_value(self, floor_strike:float,sigma:float) -> float:
        """
        Calculate the cap value of the float leg.

        Input:
        - cap (float): cap value
        - sigma (float): volatility of the forward rate

        """
        df_floor= pd.DataFrame()
        df_floor["Log"] = self._rates_c["Forward_rate"].apply(lambda x: np.log((x/100)/floor_strike))
        df_floor["vol"] = (0.5*sigma**2)*self._rates_c["Year_fraction"]
        df_floor["Actu"] = sigma*np.sqrt(self._rates_c["Year_fraction"])
        df_floor["d1"] = (df_floor["Log"]+df_floor["vol"])/df_floor["Actu"]
        df_floor["d2"] = df_floor["d1"]-df_floor["Actu"]
        df_floor["value"] = floor_strike*norm.cdf(-1*df_floor["d2"]) - self._rates_c["Forward_rate"]/100*norm.cdf(-1*df_floor["d1"])   
        self._floor_rate_dict = dict(zip(self._rates_c["Year_fraction"],  df_floor["value"]))
        self.build_cashflows(self._floor_rate_dict,1, self._cashflows_floor_r)
        self.build_cashflow_floor_npv()
        pass

    def build_cashflow_floor_npv(self):
        """
        Apply discount factors to NPV and PV01 in a cashflow dictionary.
        
        :return: New dictionary with discounted "NPV" and "PV01".
        """  
        discounted_cashflows = {
            t: {
                "NPV": cf["NPV"] * self._discount_dict.get(t, 1),
                "PV01": cf["PV01"] * self._discount_dict.get(t, 1)
            }
            for t, cf in self._cashflows_floor_r.items()
        }
        self._cashflows_floor = discounted_cashflows
        pass

class Swap(FixedIncomeProduct):
    """
    Class pour un swap classique, on va pouvoir trouver le taux d'un swap

    - Float and Fixed leg nous permettent de créer un swap.
    - La classe permet d'utiliser les fonctions de FixedLeg et FloatLeg pour trouver le taux d'un swap. 

       Input:
    - forward rates curve (dict, non optional)
    - start date (string, non optional)
    - end date (string, non optional)
    - paiments frequency (string, non optional)
    - day count convention (string, optional, equal to 30/360 if not provided)
    - rolling convention (string, optional, equal to Modified Following if not provided)
    - discounting curve to discount with a different curve than the forward rates curve (dict, optional)
    - notional (float, optional, will quote in percent if not provided)
    """
    def __init__ (self, rate_curve: Rates_curve, start_date:str, end_date:str, paiement_freq:str, currency:str, day_count:str=CONVENTION_DAY_COUNT, rolling_conv:str=ROLLING_CONVENTION, discounting_curve:Rates_curve=None, notional:float=BASE_NOTIONAL, spread:float=0, format:str=FORMAT_DATE, interpol: str=TYPE_INTERPOL, exchange_notional: str=EXCHANGE_NOTIONAL) -> None:
        super().__init__(rate_curve, start_date, end_date, paiement_freq, currency, day_count, rolling_conv, discounting_curve, notional, spread, format, interpol, exchange_notional)

        self.float_leg = FloatLeg(rate_curve, start_date, end_date, paiement_freq, currency, day_count, rolling_conv, discounting_curve, notional, spread, format, interpol, exchange_notional)

    def calculate_fixed_rate(self) -> float:
        """
        Calculate the fixed rate of the swap and initialize the FixedLeg.
        """
        # Calculer la NPV et PV01 de la jambe flottante
        float_npv = self.float_leg.calculate_npv(self.float_leg._cashflows)
        float_pv01 = self.float_leg.calculate_pv01()

        # Calculer le taux fixe
        fixed_rate = (float_npv / float_pv01) / 10000

        # Créer une copie de la courbe de taux avec le taux fixe
        self._rate_curve_fixed = self._rate_curve.deep_copy(fixed_rate)

        # Initialiser la jambe fixe
        self.fixed_leg = FixedLeg(
            self._rate_curve_fixed,
            self._start_date,
            self._end_date,
            self._paiement_freq,
            self._currency,
            self._day_count,
            self._rolling_conv,
            self._discounting_curve,
            self._notional,
            self._spread,
            self._format,
            "Flat",
            self._exchange_notional
        )

        return fixed_rate
    
    def calculate_collar(self, cap_strike:float, floor_strike:float, sigma:float) -> float:
        """
        Calculate the collar value of the swap.

        Input:
        - cap (float): cap value
        - floor (float): floor value
        - sigma (float): volatility of the forward rate
        """
        self.float_leg.cap_value(cap_strike, sigma)
        self.float_leg.floor_value(floor_strike, sigma)

        # Calculate the collar value
        collar_value = self.float_leg.calculate_npv(self.float_leg._cashflows_cap) - self.float_leg.calculate_npv(self.float_leg._cashflows_floor)
        
        return collar_value
    
    def calculate_npv(self) -> float:
        """
        Calculate the NPV of the swap.
        """
        return self.float_leg.calculate_npv() - self.fixed_leg.calculate_npv()

    def calculate_duration(self) -> float:
        """
        Calculate the duration of the swap.
        """
        return (self.float_leg.calculate_duration() + self.fixed_leg.calculate_duration()) / 2

    def calculate_sensitivity(self) -> float:
        """
        Calculate the sensitivity of the swap.
        """
        return self.float_leg.calculate_sensitivity() - self.fixed_leg.calculate_sensitivity()

    def calculate_convexity(self) -> float:
        """
        Calculate the convexity of the swap.
        """
        return self.float_leg.calculate_convexity() - self.fixed_leg.calculate_convexity()

    def calculate_pv01(self) -> float:
        """
        Calculate the PV01 of the swap.
        """
        return self.float_leg.calculate_pv01() - self.fixed_leg.calculate_pv01()

#-------------------------------------------------------------------------------------------------------
#----------------------------Classes de produits généralistes en equity derivatives---------------------
#-------------------------------------------------------------------------------------------------------
#____________________________Classe abstraite pour les produits d'Equity________________________________
class EQDProduct(ABC):
    """"
    Abstract class for equity derivatives products.

    Input:
    -Type (Enum, optional) (call / put)
    -Spot (float, optional)
    -Strike (float, optional)
    -Rate (float, optional)
    
    -date format (string, optional)
    -currency (string, optional)
    -start date (string, optional)
    -end date (string, optional) / options dates
    -day count convention (string, optional, equal to 30/360 if not provided)
    -rolling convention (string, optional, equal to Modified Following if not provided)
    -notional (float, optional, will quote in percent if not provided) / nb underlying shares / equities
    
    Returns:
    - NPV
    - Greeks: Delta / Gamma / Rho / Theta / Vega
    """
    def __init__(self, start_date:str, end_date:str, type:str=OptionType.CALL, strike:float=BASE_STRIKE, rate:float=BASE_RATE, day_count:str=CONVENTION_DAY_COUNT, rolling_conv:str=ROLLING_CONVENTION, notional:float=BASE_NOTIONAL, format_date:str=FORMAT_DATE, currency:str=BASE_CURRENCY, price:float=None) -> None:
        self._start_date = start_date
        self._end_date = end_date
        self._type = type
        self._strike = strike
        self._rate = rate
        self._day_count = day_count
        self._rolling_conv = rolling_conv
        self._notional = notional
    
        self._format=format_date
        self._currency=currency
        self._price=price

        self._paiments_schedule = \
            PaymentScheduleHandler(self._start_date, self._end_date,
            "none", self._format).build_schedule(\
            convention=self._day_count, rolling_convention=self._rolling_conv, market=\
            utils.get_market(currency=self._currency))
        
        self.T = self._paiments_schedule[-1]
        pass

    @property
    @abstractmethod
    def npv(self) -> float:
        """
        Calculate the NPV of the equity derivative product.
        """
        pass

    @property
    @abstractmethod
    def payoff(self) -> float:
        """
        Calculate the payoff of the equity derivative product.
        """
        pass

#Class for vanilla equities options:
class VanillaOption(EQDProduct):
    """"
    Abstract class for equity derivatives products.

    Input:
    -Type (Enum, optional) (call / put)
    -Spot (float, optional)
    -Strike (float, optional)
    -Rate (float, optional)
    
    -date format (string, optional)
    -currency (string, optional)
    -start date (string, optional)
    -end date (string, optional) / options dates
    -day count convention (string, optional, equal to 30/360 if not provided)
    -rolling convention (string, optional, equal to Modified Following if not provided)
    -notional (float, optional, will quote in percent if not provided) / nb underlying shares / equities
    
    Returns:
    - NPV
    - Greeks: Delta / Gamma / Rho / Theta / Vega
    """
    def __init__(self, start_date:str, end_date:str, type:str=OptionType.CALL, strike:float=BASE_STRIKE, rate:float=BASE_RATE, day_count:str=CONVENTION_DAY_COUNT, rolling_conv:str=ROLLING_CONVENTION, notional:float=BASE_NOTIONAL, format_date:str=FORMAT_DATE, currency:str=BASE_CURRENCY, div_rate:str=BASE_DIV_RATE, price:float=None, volume:float=None) -> None:
        super().__init__(start_date, end_date, type, strike, rate, day_count, rolling_conv, notional, format_date, currency, price)
        self._div_rate=div_rate
        self._volume=volume
        pass

    def npv(self, spot:float=BASE_SPOT) -> float:
        """
        Calculate the NPV of the equity derivative product.
        """
        return self.payoff(spot) * self._notional * np.exp(-self._rate * self.T)
    
    def payoff(self, spot:float=BASE_SPOT) -> float:
        """
        Calculate the payoff of the equity derivative product.
        """
        if self._type == OptionType.CALL:
            return max(spot - self._strike, 0)
        elif self._type == OptionType.PUT:
            return max(self._strike - spot, 0)
        else:
            ValueError("Option type not recognized !")
            pass
        
#Classe Action: A définir, car je sais vraiment pas quoi mettre dans celle-ci vs les EQD.
#Une possibilité serait de l'utiliser pour pricer l'action avec les modèles de diffusion, et lier un échéncier 
#de dividendes etc.
#Les inputs dans le commentaires ne sont que des directions possibles.
class Share():
    """
    Class for shares/stocks.

    Parameters:
    - spot price (float, non optional)
    - dividend schedule (dict, optional) (voir si besoin de build un échéancier comme sur le FI / genre paiement annuel des divs à partir d'une date)
    
    Returns:
    - Price of the share
    - Dividend schedule
    """
    def __init__(self) -> None:
        pass

    @property
    def spot_price(self) -> float:
        """
        Returns the spot price of the share.
        """
        return self._spot_price
    
    @property
    def schedule_dividends(self) -> dict:
        """
        Returns the dividend schedule of the share.
        """
        return self._schedule_dividends

#Classe gestion du marché des options: Ne pas bouger de fichier, obligé d'être ici pour éviter les appels en ronds.
class OptionMarket:
    """
    Class to handle the dataset of options, select appropriate ones for volatility models/parameters calibration.

    Input:
    - Filename / path for the options CSV dataset
    """
    def __init__(self, filename_options:str, filename_uderlying:str) -> None:
        self._filename = filename_options
        self._filename_underlying = filename_uderlying
        self._df_asset = pd.read_csv(self._filename_underlying, sep=';', index_col=0)
        self._df_asset.index = pd.to_datetime(self._df_asset.index)

        self._df = pd.read_csv(self._filename, sep=";")
        self._dict_df = self._split_price_dates()

        self._spot = None
        self._options_matrices = self._build_options_matrix()
        pass

    def _split_price_dates(self):
        """
        Function to split the dataset in DFs for each pricing dates found.
        """
        dfs = {}
        #Get the list of pricing dates
        pricing_dates = self._df["price_date"].unique()
        for date in pricing_dates:
            #Get the dataframe for each pricing date
            df_date = self._df[self._df["price_date"] == date].copy()
            #Get the list of columns to keep
            columns_to_keep = ["price_date", "expiration", "strike", "last", "implied_volatility", "type", "volume"]
            #Keep only the columns to keep
            df_date = df_date[columns_to_keep]
            #Add the dataframe to the dictionary
            dfs[date] = df_date
        return dfs

    def _build_options_matrix(self):
        """
        Function to build the options matrix for each pricing date.
        """
        matrices = {}
        for date in self._dict_df.keys():
            df = self._dict_df[date]
            options_matrix = defaultdict(lambda: {'call': [], 'put': []})    
            
            for _, row in df.iterrows():
                option_type = row['type'].lower()
                if option_type == "call":
                    t = 1
                elif option_type == "put":
                    t = -1
                else:
                    print(f"Invalid Option type: {option_type}")

                option = VanillaOption(row['price_date'], row['expiration'], OptionType(t), row['strike'], price=row['last'], volume=row['volume'])
                options_matrix[row['expiration']][option_type].append(option)
            
            matrices[date]=options_matrix
        return matrices

    def get_options_for_moneyness(self, price_date:str, maturity:str, moneyness_bounds:tuple=BOUNDS_MONEYNESS, calibrate_on_OTM:bool=OTM_CALIBRATION, calibrate_on_volume:bool=VOLUME_CALIBRATION, volume_bounds:float=VOLUME_THRESHOLD) -> list:
        """
        Function to get options in a certains moneyness for a specified pricing date, and specified maturity.

        Input:
        - spot (float, non optional): spot price of the underlying asset
        - price_date (string, optional): price date of the options
        - maturity (string, optional): maturity date of the options
        - moneyness_bounds (tuple, optional): tuple of min and max moneyness to filter the options (in % or 0.00 format)
        """
        #Getting the spot
        self._spot = self._df_asset.loc[pd.to_datetime(price_date),"4. close"]

        #Taking correct options
        options_for_dates = self._options_matrices[price_date][maturity]
        options_for_moneyness = []
        k_moneyness = lambda strike:np.log(strike/self._spot)
        if calibrate_on_volume == True:
            count, volume = 0, 0
            for option in options_for_dates['call']:
                count+=1
                volume+=option._volume
            for option in options_for_dates['put']:
                count+=1
                volume+=option._volume
            volume/=count

            if calibrate_on_OTM == True:
                for option in options_for_dates['call']:
                    if k_moneyness(option._strike) >= np.log(moneyness_bounds[0]) and k_moneyness(option._strike) <= np.log(moneyness_bounds[1]) and k_moneyness(option._strike) >= 0 and option._volume>volume_bounds*volume:
                        options_for_moneyness.append(option)
                for option in options_for_dates['put']:
                    if k_moneyness(option._strike) >= np.log(moneyness_bounds[0]) and k_moneyness(option._strike) <= np.log(moneyness_bounds[1]) and k_moneyness(option._strike) <= 0 and option._volume>volume_bounds*volume:
                        options_for_moneyness.append(option)
            else:
                for option in options_for_dates['calls']:
                    if  k_moneyness(option._strike) >= np.log(moneyness_bounds[0]) and k_moneyness(option._strike) <= np.log(moneyness_bounds[1]) and option._volume>volume_bounds*volume:
                        options_for_moneyness.append(option)
                for option in options_for_dates['puts']:
                    if  k_moneyness(option._strike) >= np.log(moneyness_bounds[0]) and k_moneyness(option._strike) <= np.log(moneyness_bounds[1]) and option._volume>volume_bounds*volume:
                        options_for_moneyness.append(option)
        
        else:
            if calibrate_on_OTM == True:
                for option in options_for_dates['call']:
                    if k_moneyness(option._strike) >= np.log(moneyness_bounds[0]) and k_moneyness(option._strike) <= np.log(moneyness_bounds[1]) and k_moneyness(option._strike) >= 0:
                        options_for_moneyness.append(option)
                for option in options_for_dates['put']:
                    if k_moneyness(option._strike) >= np.log(moneyness_bounds[0]) and k_moneyness(option._strike) <= np.log(moneyness_bounds[1]) and k_moneyness(option._strike) <= 0:
                        options_for_moneyness.append(option)
            else:
                for option in options_for_dates['calls']:
                    if  k_moneyness(option._strike) >= np.log(moneyness_bounds[0]) and k_moneyness(option._strike) <= np.log(moneyness_bounds[1]):
                        options_for_moneyness.append(option)
                for option in options_for_dates['puts']:
                    if  k_moneyness(option._strike) >= np.log(moneyness_bounds[0]) and k_moneyness(option._strike) <= np.log(moneyness_bounds[1]):
                        options_for_moneyness.append(option)
        return options_for_moneyness
    
    def get_values_for_calibration_SVI(self, price_date:str, maturity:str, moneyness_bounds:tuple=BOUNDS_MONEYNESS, calibrate_on_OTM:bool=OTM_CALIBRATION, calibrate_on_volume:bool=VOLUME_CALIBRATION, volume_bounds:float=VOLUME_THRESHOLD) -> list:
        """
        Function to get the values for the calibration of the SSI model.

        Input:
        - spot (float, non optional): spot price of the underlying asset
        - price_date (string, optional): price date of the options
        - maturity (string, optional): maturity date of the options
        - moneyness_bounds (tuple, optional): tuple of min and max moneyness to filter the options (in % or 0.00 format)
        """
        options = self.get_options_for_moneyness(price_date, maturity, moneyness_bounds, calibrate_on_OTM, calibrate_on_volume, volume_bounds)
        return [option._type for option in options], [option._strike for option in options], [option._price for option in options], self._spot, options[0].T, options

#Classe de calibration SSVI:
class SSVICalibration:
    """
    Class to calibrate SSVI parameters for a given underlying stock, based on SVI calibration ATM for different maturities.
    Repricing the ATMs options for each maturity, and calibrating the SSVI parameters for ATM options.
    Finally, calibrating the whole SSVI equations to get a full Volatility Surface.
    Be carefull to pass only options for the pricings, need to be cleaned on wanted moneyness levels and liquidity.
    Input:
    - model: Model used for options pricing (we advise BSM for fast executions).
    - data_path: path to the options dataset.
    - file_name_underlying: path of the underlying asset file.
    - pricing_date: date of the desired SSVI.
    - moneyness_level: tuple of min and max moneyness to filter the options (in % or 0.00 format, optional).
    - OTM_calibration: boolean to calibrate on OTM options or not (optional).

    Returns:
    - params: dictionary of SSVI parameters for all maturities.    
    """
    def __init__(self, model:str, data_path:str, file_name_underlying:str, pricing_date:str, moneyness_level:tuple=BOUNDS_MONEYNESS, OTM_calibration:bool=OTM_CALIBRATION, div_rate:float=BASE_DIV_RATE, currency:str=BASE_CURRENCY, rate:float=BASE_RATE, initial_ssvi:list=INITIAL_SSVI, ssvi_method:str=SSVI_METHOD, ssvi_options:dict=OPTIONS_SOLVER_SSVI) -> None:
        self._model = model
        self._model_obj = dict_models[model]
        self._pricing_date = pricing_date
        self._option_market = OptionMarket(data_path, file_name_underlying)
        self._maturities = list(self._option_market._options_matrices[self._pricing_date].keys())
        self._moneyness_level = moneyness_level
        self._OTM_calibration = OTM_calibration
        self._initial_params_ssvi = initial_ssvi
        self._ssvi_method = ssvi_method
        self._ssvi_options = ssvi_options
        self._options = self._get_market_options()
        #self._market_prices = np.array([option._price for option in self._options])

        self._div_rate = div_rate
        self._currency = currency
        self._rate = rate
        self._spot = None

        self._options_for_calibration = None
        self._maturities_t = {}
        self._params = self._params_svis
        self._calibration_prices = np.array([option._price for option in self._options_for_calibration])
        self._atm_options = {}
        self._atm_prices = self._reprice_ATM_options

    @property
    def _params_svis(self)->dict:
        """
        Returns a dict of SVI parameters for all the maturities for a given dict of options (equal one maturity).
        """
        params = {}
        res_options_for_calibration = []
        for maturity_date in self._maturities:
            list_types, list_strikes, list_prices, self._spot, t_options, options_for_calibration = self._option_market.get_values_for_calibration_SVI(self._pricing_date, maturity_date, self._moneyness_level , self._OTM_calibration)
            if list_types is not None:
                pricer = OptionPricer(self._pricing_date, maturity_date, model=self._model, spot=self._spot, div_rate=self._div_rate, currency=self._currency, rate=self._rate, notional=1)
                params[maturity_date]= pricer.svi_params(list_types, list_strikes, list_prices)
                res_options_for_calibration.extend(options_for_calibration)
                self._maturities_t[maturity_date]=t_options
        
        self._options_for_calibration = res_options_for_calibration
        return params

    @property
    def _reprice_ATM_options(self)->dict:
        """
        Use the params to price ATM implied volatility for each maturity and use those to calibrate phi function of SSVI.
        """
        prices = {}
        for maturity_date in self._maturities:
            params = self._params[maturity_date]
            if params is not None: #np.log(1) = 0
                atm_vol = np.sqrt((params[0] + params[1] * (params[2]*(-params[3]) + np.sqrt((-params[3])**2 + params[4]**2)))/self._maturities_t[maturity_date])
                pricer = OptionPricer(self._pricing_date, maturity_date, OptionType.CALL, self._model, self._spot, self._spot, self._div_rate, rate=self._rate, sigma=atm_vol)
                prices[self._maturities_t[maturity_date]] = pricer.price
                self._atm_options[maturity_date] = pricer.get_option()
        return prices

    def _calibrate_theta(self)->list:
        """
        Used to calibrate phi function of the SSVI.
        """
        fct_theta=lambda k,v_o,v_inf,t: (((1-np.exp(-k*t))/(k*t))*(v_o-v_inf)+v_inf)*t
        fct_valo=lambda option, theta: self._model_obj(option, theta)

        def new_initial(bounds):
            perturbation = np.ones(len(self._initial_params_ssvi[0:3])) * np.random.uniform(-5,5)
            new_init = self._initial_params_ssvi[0:3]*perturbation
            for i in range(len(bounds)):
                if bounds[i][1] == None:
                    bounds[i][1] = 1000000
                if bounds[i][0] == None:    
                    bounds[i][0] = -1000000

            if all(new_init[i] < bounds[i][1] and new_init[i] > bounds[i][0] for i in range(len(self._initial_params_ssvi[0:3]))):
                return new_init
            else:
                return new_initial(bounds)

        def objective(params)->float:
            """
            Objective function to minimize.
            """
            k, v_o, v_inf = params
            maturities = np.array(list(self._maturities_t.values()))
            theta_vec = np.sqrt(fct_theta(k,v_o,v_inf, maturities)/maturities)
            prices = [fct_valo(option, theta).price(self._spot) for option, theta in zip(list(self._atm_options.values()), theta_vec)]
            return np.sum((np.array(prices)-np.array(list(self._atm_prices.values())))**2)
        
        bounds = ([None,None], [1e-6,None], [1e-6,None]) #((1e-16,10), (1e-6,5), (1e-6,5))
        result = minimize(objective, self._initial_params_ssvi[0:3], bounds=bounds, method=self._ssvi_method, options=self._ssvi_options)

        if result.success and result["fun"]<20:
            return result.x
        elif result.success and result["fun"]>20:
            self._initial_params_ssvi[0:3]=new_initial(bounds)
            return self._calibrate_theta()
        else:
            print(f"Optimization failed.") #Method: {self._method}, Tolerance: {self._tolerance}, Max Iterations: {count}, Bounds: {self._bounds}, Starting Point: {self._starting_point}")
            return None
        
    def calibrate_SSVI(self)->dict:
        """
        Calibrate the SSVI parameters for all maturities.
        """
        k, v_0, v_inf = self._calibrate_theta()
        fct_theta=lambda k,v_o,v_inf,t: (((1-np.exp(-k*t))/(k*t))*(v_o-v_inf)+v_inf)*t
        fct_prices=lambda option, vol: self._model_obj(option,vol).price(self._spot)

        def ssvi_total_variance(thetas, k_vec, rho, mu, l):
            """
            SSVI total variance function.
            """
            phi = mu*(thetas **l)
            term = phi * k_vec + rho
            sqrt_term = np.sqrt(term**2 + (1 - rho**2))
            return 0.5 * thetas * (1 + rho * phi * k_vec + sqrt_term)
        
        vec_maturities = np.array([option.T for option in self._options_for_calibration])
        vec_thetas = np.array(fct_theta(k,v_0,v_inf, vec_maturities))
        vec_k = np.array([np.log(option._strike/self._spot) for option in self._options_for_calibration])

        def objective(params)->float:
            """
            Objective function to minimize.
            """
            rho, mu, l = params
            vec_w = ssvi_total_variance(vec_thetas, vec_k, rho, mu, l)
            ssvi_vol = np.sqrt(vec_w/vec_maturities)
            prices = [fct_prices(option, vol) for option, vol in zip(self._options_for_calibration, ssvi_vol)]
            return np.sum((np.array(prices)-self._calibration_prices)**2)

        bounds = ((-0.9999,0.9999), (1e-16, None), (1e-16, 1))
        result = minimize(objective, self._initial_params_ssvi[-3:], bounds=bounds, method=self._ssvi_method, options=self._ssvi_options)
        if result.success:
            rho, mu, l = result.x
            params = {"K": k, "v_0": v_0, "v_inf": v_inf, "rho": rho, "mu": mu, "l": l}
            return params
        else:
            print(f"Calibration failed.")
            return None
    
    def _get_market_options(self)->dict:
        """
        Returns the options market.
        """
        list_options = []
        for maturity in self._maturities:
            for option in self._option_market._options_matrices[self._pricing_date][maturity]['call']:
                list_options.append(option)
            for option in self._option_market._options_matrices[self._pricing_date][maturity]['put']:
                list_options.append(option)

        return list_options

#Classe SSVI:
class SSVICalibration2:
    """
    Class to calibrate SSVI parameters for a given underlying stock, based on SVI calibration ATM for different maturities.
    Repricing the ATMs options for each maturity, and calibrating the SSVI parameters for ATM options.
    Finally, calibrating the whole SSVI equations to get a full Volatility Surface.
    Be carefull to pass only options for the pricings, need to be cleaned on wanted moneyness levels and liquidity.
    Input:
    - model: Model used for options pricing (we advise BSM for fast executions).
    - data_path: path to the options dataset.
    - file_name_underlying: path of the underlying asset file.
    - pricing_date: date of the desired SSVI.
    - moneyness_level: tuple of min and max moneyness to filter the options (in % or 0.00 format, optional).
    - OTM_calibration: boolean to calibrate on OTM options or not (optional).

    Returns:
    - params: dictionary of SSVI parameters for all maturities.    
    """
    def __init__(self, model:str, data_path:str, file_name_underlying:str, pricing_date:str, moneyness_level:tuple=BOUNDS_MONEYNESS, OTM_calibration:bool=OTM_CALIBRATION, div_rate:float=BASE_DIV_RATE, currency:str=BASE_CURRENCY, rate:float=BASE_RATE, initial_ssvi:list=INITIAL_SSVI, ssvi_method:str=SSVI_METHOD, ssvi_options:dict=OPTIONS_SOLVER_SSVI) -> None:
        self._model = model
        self._model_obj = dict_models[model]
        self._pricing_date = pricing_date
        self._option_market = OptionMarket(data_path, file_name_underlying)
        self._maturities = list(self._option_market._options_matrices[self._pricing_date].keys())
        self._moneyness_level = moneyness_level
        self._OTM_calibration = OTM_calibration
        self._initial_params_ssvi = initial_ssvi
        self._ssvi_method = ssvi_method
        self._ssvi_options = ssvi_options
        self._options = self._get_market_options()
        self._market_prices = np.array([option._price for option in self._options])

        self._div_rate = div_rate
        self._currency = currency
        self._rate = rate
        self._spot = None

        self._maturities_t = {}
        self._params = self._params_svis
        self._thetas=self.calculate_thetas()
        print(self._thetas)
        pass

    @property
    def _params_svis(self)->dict:
        """
        Returns a dict of SVI parameters for all the maturities for a given dict of options (equal one maturity).
        """
        params = {}
        for maturity_date in self._maturities:
            list_types, list_strikes, list_prices, self._spot, t_options = self._option_market.get_values_for_calibration_SVI(self._pricing_date, maturity_date, self._moneyness_level , self._OTM_calibration)
            if list_types is not None:
                pricer = OptionPricer(self._pricing_date, maturity_date, model=self._model, spot=self._spot, div_rate=self._div_rate, currency=self._currency, rate=self._rate, notional=1)
                params[maturity_date]= pricer.svi_params(list_types, list_strikes, list_prices)
                self._maturities_t[maturity_date]=t_options
        return params
    
    def calculate_thetas(self)->dict:
        """
        Calculate the theta of the SSVI parameters for all maturities.
        """
        t_vec = np.array(list(self._maturities_t.values()))
        params_array = np.array(list(self._params.values()))
        a = params_array[:, 0]
        b = params_array[:, 1]
        rho = params_array[:, 2]
        m = params_array[:, 3]
        s = params_array[:, 4]
        thetas = a + b * (rho * (-m) + np.sqrt((m)**2 + s**2))
        return dict(zip(t_vec, thetas))

    def _get_market_options(self)->dict:
        """
        Returns the options market.
        """
        list_options = []
        for maturity in self._maturities:
            for option in self._option_market._options_matrices[self._pricing_date][maturity]['call']:
                list_options.append(option)
            for option in self._option_market._options_matrices[self._pricing_date][maturity]['put']:
                list_options.append(option)

        return list_options

#-------------------------------------------------------------------------------------------------------
#----------------------------Script pour implémenter les différentes classes prices---------------------
#-------------------------------------------------------------------------------------------------------

#Options pricer:
class OptionPricer:
    """
    Class for pricing options using different models.

    Input:
    - start_date: Start date of the option in the format.
    - end_date: End date of the option in the format.
    - type: Type of the option (CALL or PUT).
    - model: Model used for pricing (optional).
    - spot: Spot price of the underlying asset (optional).
    - strike: Strike price of the option (optional).
    - div_rate: Dividend rate (optional).
    - day_count: Day count convention (optional).
    - rolling_conv: Rolling convention (optional).
    - notional: Notional amount (optional).
    - format_date: Date format (optional).
    - currency: Currency of the option (optional).
    - sigma: Implied volatility (optional).
    - rate: Risk-free interest rate (optional).
    """
    def __init__(self, start_date:str, end_date:str, type:str=OptionType.CALL, model:str=BASE_MODEL, spot:float=BASE_SPOT, strike:float=BASE_STRIKE, div_rate:float=BASE_DIV_RATE, day_count:str=CONVENTION_DAY_COUNT, rolling_conv:str=ROLLING_CONVENTION, notional:float=BASE_NOTIONAL, format_date:str=FORMAT_DATE, currency:str=BASE_CURRENCY, sigma:float=None, rate:float=BASE_RATE, price:float=None) -> None:
        self._model = model
        self._start_date = start_date
        self._end_date = end_date
        self._type = type
        self._spot = spot
        self._strike = strike
        self._div_rate = div_rate
        self._day_count = day_count
        self._rolling_conv = rolling_conv
        self._notional = notional
        self._format_date = format_date
        self._currency = currency
        self._sigma = sigma
        self._rate = rate

        self.payoff = None
        self._price = price

    @property
    def price(self):
        if self._model == "Black-Scholes-Merton":
            if self._sigma is None:
                self._sigma = BASE_SIGMA
            
            self._option = VanillaOption(start_date=self._start_date, end_date=self._end_date, type=self._type, strike=self._strike, notional=self._notional, currency=self._currency, div_rate=self._div_rate)
            self._model = dict_models[self._model](self._option, self._sigma)
            price = self._model.price(self._spot)
            self.payoff = price * np.exp(self._rate * self._option.T) * self._notional
            return price
        
    @property
    def delta(self):
        if self._model == "Black-Scholes-Merton":
            if self._sigma is None:
                self._sigma = BASE_SIGMA
            
            self._option = VanillaOption(start_date=self._start_date, end_date=self._end_date, type=self._type, strike=self._strike, notional=self._notional, currency=self._currency, div_rate=self._div_rate)
            self._model = dict_models[self._model](self._option, self._sigma)
            return self._model.delta(self._spot)

    @property
    def gamma(self):
        if self._model == "Black-Scholes-Merton":
            if self._sigma is None:
                self._sigma = BASE_SIGMA
            
            self._option = VanillaOption(start_date=self._start_date, end_date=self._end_date, type=self._type, strike=self._strike, notional=self._notional, currency=self._currency, div_rate=self._div_rate)
            self._model = dict_models[self._model](self._option, self._sigma)
            return self._model.gamma(self._spot)

    @property
    def vega(self):
        if self._model == "Black-Scholes-Merton":
            if self._sigma is None:
                self._sigma = BASE_SIGMA
            
            self._option = VanillaOption(start_date=self._start_date, end_date=self._end_date, type=self._type, strike=self._strike, notional=self._notional, currency=self._currency, div_rate=self._div_rate)
            self._model = dict_models[self._model](self._option, self._sigma)
            return self._model.vega(self._spot)

    @property
    def theta(self):
        if self._model == "Black-Scholes-Merton":
            if self._sigma is None:
                self._sigma = BASE_SIGMA
            
            self._option = VanillaOption(start_date=self._start_date, end_date=self._end_date, type=self._type, strike=self._strike, notional=self._notional, currency=self._currency, div_rate=self._div_rate)
            self._model = dict_models[self._model](self._option, self._sigma)
            return self._model.theta(self._spot, self._rate)

    @property
    def rho(self):
        if self._model == "Black-Scholes-Merton":
            if self._sigma is None:
                self._sigma = BASE_SIGMA
            
            self._option = VanillaOption(start_date=self._start_date, end_date=self._end_date, type=self._type, strike=self._strike, notional=self._notional, currency=self._currency, div_rate=self._div_rate)
            self._model = dict_models[self._model](self._option, self._sigma)
            return self._model.rho(self._spot, self._rate)

    def implied_vol(self, method:str=BASE_METHOD_VOL, tolerance:float=TOLERANCE, max_iter:float=MAX_ITER, bounds:tuple=BOUNDS, starting_point:float=STARTING_POINT) -> float:
        if self._price is None:
            self._price = self.price
        if self._model == "Black-Scholes-Merton":
            self._option = VanillaOption(start_date=self._start_date, end_date=self._end_date, type=self._type, strike=self._strike, notional=self._notional, currency=self._currency, div_rate=self._div_rate, price=self._price)
            self._model = dict_models[self._model]
            volatility_finder = ImpliedVolatilityFinder(model=self._model, option=self._option, price=self._price, method=method, tolerance=tolerance, nb_iter=max_iter, bounds=bounds, starting_point=starting_point, spot=self._spot)
            return volatility_finder.find_implied_volatility()
        else:
            raise ValueError("Model not supported for implied volatility calculation. Please choose Black-Scholes-Merton.")
        
    def svi_params(self, vector_types:list, vector_strikes:list, vector_prices:list, method:str=BASE_METHOD_VOL, tolerance:float=TOLERANCE, max_iter:float=MAX_ITER, bounds:tuple=BOUNDS, starting_point:float=STARTING_POINT) -> tuple:
        if self._model == "Black-Scholes-Merton":
            options=[]
            for i in range(len(vector_strikes)):
                options.append(VanillaOption(start_date=self._start_date, end_date=self._end_date, type=vector_types[i], strike=vector_strikes[i], notional=self._notional, currency=self._currency, div_rate=self._div_rate))
            self._model = dict_models[self._model]
            svi_params_finder = SVIParamsFinder(model=self._model, vector_options=options, vector_prices=vector_prices, method_implied_vol=method, spot=self._spot, tolerance=tolerance, nb_iter=max_iter, bounds=bounds, starting_point=starting_point)
            result = svi_params_finder.find_svi_parameters()
            return result
        else:
            raise ValueError("Didn't find SVI Parameters with these inputs!")

    def get_option(self):
        return VanillaOption(start_date=self._start_date, end_date=self._end_date, type=self._type, strike=self._strike, notional=self._notional, currency=self._currency, div_rate=self._div_rate)