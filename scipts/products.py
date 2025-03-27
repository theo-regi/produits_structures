from constants import BASE_NOTIONAL, CONVENTION_DAY_COUNT, ROLLING_CONVENTION, FORMAT_DATE, TYPE_INTERPOL, EXCHANGE_NOTIONAL, BASE_SHIFT

import numpy as np
from abc import ABC, abstractmethod
from utils import Maturity_handler, PaymentScheduleHandler, Rates_curve
import utils
import pandas as pd
from scipy.stats import norm

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
    def __init__(self, rate_curve: Rates_curve, start_date:str, end_date:str,
                 paiement_freq:str, currency:str, day_count:str=CONVENTION_DAY_COUNT, rolling_conv:str=ROLLING_CONVENTION,
                 discounting_curve:Rates_curve=None, notional:float=BASE_NOTIONAL, spread:float=0, format:str=FORMAT_DATE, interpol: str=TYPE_INTERPOL, exchange_notional: str=EXCHANGE_NOTIONAL) -> None:
        
        self._rate_curve=rate_curve
        self._start_date=start_date
        self._end_date=end_date
        self._paiement_freq=paiement_freq
        self._currency=currency
        self._day_count=day_count
        self._rolling_conv=rolling_conv
        self._notional = notional
        self._format = format
        self._cashflows = {}
        self._cashflows_cap ={}
        self._cashflows_r = {}
        self._cashflows_cap_r ={}
        self._exchange_notional = exchange_notional
        self._spread = spread
        self._interpol = interpol
        self._paiments_schedule = \
            PaymentScheduleHandler(self._start_date, self._end_date,
            self._paiement_freq, self._format).build_schedule(\
            convention=self._day_count, rolling_convention=self._rolling_conv, market=\
            utils.get_market(currency=self._currency))

    @abstractmethod
    def calculate_npv(self) -> float:
        """
        Returns the product NPV as float
        """
        return sum(entry["NPV"] for entry in self._cashflows.values())
    
    @abstractmethod
    def calculate_duration(self) -> float:
        """
        Returns duration of the product
        """
        duration_ti = sum(value["NPV"] * key for key, value in self._cashflows.items())
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
        return sum(entry["PV01"] for entry in self._cashflows.values())

#Fixed Leg Income product
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

    def calculate_sensitivity(self, shift:dict=None) -> float:
        """
        Calculate the sensitivity of the fixed leg.

        Input:
        - shift (dict, optionnal): dictionnary of shift for each date, if not given -> linear shift of 1bps.
        """
        if shift is None:
            s = np.ones(len(self._paiments_schedule)) * BASE_SHIFT
            shift = dict(zip(self._paiments_schedule, s))
        shifted_curve = self._discounting_c.deep_copy()
        shifted_curve.shift_curve(shift, self._interpol)
        shift_fixed_leg = FixedLeg(self._rate_curve, self._start_date, self._end_date, self._paiement_freq, self._currency, self._day_count, self._rolling_conv, shifted_curve, self._notional, self._spread, self._format, self._interpol, self._exchange_notional)
        shift_fixed_leg.calculate_npv()
        return shift_fixed_leg.calculate_npv() - self.calculate_npv()

    def calculate_convexity(self, shift:dict=None) -> float:
        """
        Calculate the convexity of the fixed leg.

        Input:
        - shift (dict, optionnal): dictionnary of shift for each date, if not given -> linear shift of 1bps (0.01 input).
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
        return sum((shift_leg_pos.calculate_npv() + shift_leg_neg.calculate_npv() - 2 * self.calculate_npv()) /
            ((shift[t]/100 ** 2) * self.calculate_npv()) for t in self._paiments_schedule)
    
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
        - exchange_notionnal (string, optionnal, equal to False if not provided), provide True for bonds.
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
    - forward rates curve (dict, non optionnal)
    - start date (string, non optionnal)
    - end date (string, non optionnal)
    - paiments frequency (string, non optionnal)
    - day count convention (string, optionnal, equal to 30/360 if not provided)
    - rolling convention (string, optionnal, equal to Modified Following if not provided)
    - discounting curve to discount with a different curve than the forward rates curve (dict, optionnal)
    - notional (float, optionnal, will quote in percent if not provided)
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
    
    def calculate_sensitivity(self, shift_fw:dict=None, shift_discounting:dict=None) -> float:
        """
        Calculate the sensitivity of the float leg.

        Input:
        - shift_fw (dict, optionnal): dictionnary of shift for each date, if not given -> linear shift of 1bps.
        - shift_discounting (dict, optionnal): dictionnary of shift for each date, if not given -> linear shift of 1bps.
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
        shift_fixed_leg.calculate_npv()
        return shift_fixed_leg.calculate_npv() - self.calculate_npv()

    def calculate_convexity(self, shift_fw:dict=None, shift_discounting:dict=None) -> float:
        """
        Calculate the convexity of the float leg.

        Input:
        - shift_fw (dict, optionnal): dictionnary of shift for each date, if not given -> linear shift of 1bps.
        - shift_discounting (dict, optionnal): dictionnary of shift for each date, if not given -> linear shift of 1bps.
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
            return shift_leg.calculate_npv()

        #Initial NPV:
        npv_0 = self.calculate_npv()

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
        - exchange_notionnal (string, optionnal, equal to False if not provided), provide True for bonds.
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
        print(self._cap_rate_dict)
        self.build_cashflows(self._cap_rate_dict,1, self._cashflows_cap_r)
        self.build_cashflow_cap_npv()
        return df_cap["value"]

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
 
"""
1: On va utiliser cette classe abstraite pour tout les produits composés de ZC:
- Float et Fix leg, car ce seront des bases pour construire le reste, il faut un input échange de notionnel (yes/no)
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

class Swap(FixedIncomeProduct):
    """
    Class pour un swap classique, on va pouvoir trouver le taux d'un swap

    - Float and Fixed leg nous permettent de créer un swap.
    - La classe permet d'utiliser les fonctions de FixedLeg et FloatLeg pour trouver le taux d'un swap. 

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
    def __init__ (self, rate_curve: Rates_curve, start_date:str, end_date:str, paiement_freq:str, currency:str, day_count:str=CONVENTION_DAY_COUNT, rolling_conv:str=ROLLING_CONVENTION, discounting_curve:Rates_curve=None, notional:float=BASE_NOTIONAL, spread:float=0, format:str=FORMAT_DATE, interpol: str=TYPE_INTERPOL, exchange_notional: str=EXCHANGE_NOTIONAL) -> None:
        super().__init__(rate_curve, start_date, end_date, paiement_freq, currency, day_count, rolling_conv, discounting_curve, notional, spread, format, interpol, exchange_notional)

        self.float_leg = FloatLeg(rate_curve, start_date, end_date, paiement_freq, currency, day_count, rolling_conv, discounting_curve, notional, spread, format, interpol, exchange_notional)
        self.fixed_rate = self.calculate_fixed_rate()

        # Create a fixed rate curve for the FixedLeg
        self._rate_curve_fixed=rate_curve.deep_copy(self.fixed_rate)
        self.fixed_leg = FixedLeg(self._rate_curve_fixed, start_date, end_date, paiement_freq, currency, day_count, rolling_conv, discounting_curve, notional,spread, format, "Flat", exchange_notional)


    def calculate_fixed_rate(self) -> float:
        """
        Calculate the fixed rate of the swap.
        """
        float_npv = self.float_leg.calculate_npv()
        float_pv01 = self.float_leg.calculate_pv01()
        fixed_rate = (float_npv / float_pv01) / 10000
        return fixed_rate
    
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
#Classe Action: A définir, car je sais vraiment pas quoi mettre dans celle-ci vs les EQD.
#Une possibilité serait de l'utiliser pour pricer l'action avec les modèles de diffusion, et lier un échéncier 
#de dividendes etc.
#Les inputs dans le commentaires ne sont que des directions possibles.
class Share():
    """
    Class for shares/stocks.

    Input:
    - spot price (float, non optionnal)
    - dividend schedule (dict, optionnal) (voir si besoin de build un échéancier comme sur le FI / genre paiement annuel des divs à partir d'une date)
    - diffusion model
    - risk free rate curve to calculate forward price + discounting
    
    Returns:
    - Price of the share
    - Dividend schedule
    - Forward price at risk free rate
    """
    def __init__(self, spot_price: float) -> None:
        pass

#____________________________Classe abstraite pour les produits d'Equity________________________________
class EQDProduct(ABC):
    """"
    Abstract class for equity derivatives products.

    Input:
    -underlying equity (class, non optionnal)
    -forward rates curve (dict, non optionnal) ?
    -start date (string, non optionnal)
    -end date (string, non optionnal) / options dates
    -paiments frequency (string, non optionnal) (for divs par exemple/coupon pour des reverses convertibles / jsais pas)
    -day count convention (string, optionnal, equal to 30/360 if not provided)
    -rolling convention (string, optionnal, equal to Modified Following if not provided)
    -discounting curve to discount with a different curve than the forward rates curve (dict, optionnal)
    -notional (float, optionnal, will quote in percent if not provided) / nb underlying shares / equities

    Returns:
    - NPV
    - Greeks: Delta / Gamma / Rho / Theta / Vega
    - Sensitivities : VaR / CVaR ?
    """
    def __init__(self):
        pass

#Class for vanilla equities options:
class VanillaOption(EQDProduct):
    """
    Class for vanilla options on equities."

    Input:
    - underlying equity (class, non optionnal)
    - start date (string, non optionnal)
    - end date (string, non optionnal) / options dates
    - type of option (string, non optionnal) (call / put)
    - strike price (float, non optionnal)
    - day count convention (string, optionnal, equal to 30/360 if not provided)
    - rolling convention (string, optionnal, equal to Modified Following if not provided)
    - discounting curve to discount with a different curve than the forward rates curve (dict, optionnal)
    - notional (float, optionnal, will quote in percent if not provided) / nb underlying shares / equities
    - type of exercise (string, optionnal) (American / European / Bermudan ?)

    Returns:
    - NPV
    - Greeks: Delta / Gamma / Rho / Theta / Vega
    - Sensitivities : VaR / CVaR ?
    """
    def __init__(self):
        pass

"""
Après à quel point on veut aller dans la complexité des produits/ou avoir des produits sur mesures.
Est-ce qu'on veut des produits structurés (reverse convertibles, autocallables, etc.) ou des produits plus
simples (options, futures, etc.)

Est-ce qu'on peut utiliser direct la classe option pour faire des barrières, des digitals, etc.
"""