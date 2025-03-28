from constants import CONVENTION_DAY_COUNT, TYPE_INTERPOL, INITIAL_RATE
from datetime import datetime as dt
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import holidays
import pandas as pd
import numpy as np
from scipy.optimize import fsolve

from functions import optimize_nelson_siegel,nelson_siegel
#-------------------------------------------------------------------------------------------------------
#----------------------------Script pour implémenter les classes utilitaires----------------------------
#-------------------------------------------------------------------------------------------------------

#Maturity handler :: convention, format, rolling convention, market -> year fraction
class Maturity_handler:
    """
    Classe pour gérer les différents problèmes de maturités -> renvoie une year_fraction
    
    Input: 
        - une convention de day count (30/360, etc) en string
        - un format de date (%d/%m/%Y = 01/01/2025)
        - une rolling convention = comment on gère les jours de marchés fermés (Modified Following, etc)
        - un marché (UE = XECB, US = XNYS, Brésil/Lattam = BVMF, UK = IFEU), (ce n'est pas optimal car pas sur que les jours fériés soient corrects, il faudrait une fonction bloomberg, mais inaccessible hors de l'université)

    Pour l'utiliser, appeller get_year_fraction, avec en input, une date proche, une date éloignée.
    
    Renvoie un float = nombres d'années, 1.49 = +/- 1 ans et 6 mois (fonction des jours fériés, convention, etc).
    """
    def __init__(self, convention: str, format_date: str, rolling_convention: str, market: str) -> None:
        self.__convention = convention
        self.__format_date = format_date
        self.__rolling_convention = rolling_convention
        self.__market = market
        self.__calendar = self.__get_market_calendar(dt.today()-timedelta(days=100*360), dt.today()+timedelta(days=100*360))
        pass

    def __convention_handler(self, valuation_date, end_date) -> float:
        """Returns the corresponding year_fraction (end_date - valuation_date)
            corresponding to the convention of the handler."""
        d1, m1, y1 = valuation_date.day, valuation_date.month, valuation_date.year
        d2, m2, y2 = end_date.day, end_date.month, end_date.year
        if d1 == 31:
            d1 = 30
        if d2 == 31:
            d2 = 30
        if self.__convention == "30/360":
            return (360*(y2 - y1) + 30 * (m2 - m1) + (d2 - d1))/360
        elif self.__convention == "Act/360":
            delta_days  = (end_date - valuation_date).days
            return delta_days/360
        elif self.__convention == "Act/365":
            delta_days  = (end_date - valuation_date).days
            return delta_days/365
        elif self.__convention == "Act/Act":
            days_count = 0
            current_date = valuation_date
            while current_date < end_date:
                year_end = dt(current_date.year, 12, 31)
                if year_end > end_date:
                    year_end = end_date
                days_in_year = (dt(current_date.year, 12, 31) - dt(current_date.year, 1, 1)).days + 1
                days_count += (year_end - current_date).days / days_in_year
                current_date = year_end + timedelta(days=1)
            return days_count
        else:
            raise ValueError(f"Entered Convention: {self.__convention} is not handled ! (30/360, Act/360, Act/365, Act/Act)")

    def __get_market_calendar(self, start_date, end_date):
        try:
            return holidays.financial_holidays(market=self.__market, years=(range(start_date.year, end_date.year)))
        except:
            raise ValueError(f"Error calendar: {self.__market} is not supported Choose (XECB, IFEU, XNYS, BVMF)")
        pass
        
    def __get_next_day(self, date):
        while date.weekday() >= 5 or date in self.__calendar:
            date += timedelta(days=1)
        return date
    
    def __get_previous_day(self, date):
        while date.weekday() >= 5 or date in self.__calendar:
            date -= timedelta(days=1)
        return date

    def __apply_rolling_convention(self, date):
        if self.__rolling_convention == "Following":
            return self.__get_next_day(date)
        
        elif self.__rolling_convention == "Modified Following":
            new_date = self.__get_next_day(date)
            if new_date.month != date.month:
                return self.__get_previous_day(date)
            else:
                return new_date
            
        elif self.__rolling_convention == "Preceding":
            return self.__get_previous_day(date)
        
        elif self.__rolling_convention == "Modified Preceding":
            new_date = self.__get_previous_day(date)
            if new_date.month != date.month:
                return self.__get_next_day(date)
            else:
                return new_date
        else:
            raise ValueError(f"Rolling Convention {self.__rolling_convention} is not supported ! Choose: Following, Modified Following, Preceding, Modified Preceding")

    def get_year_fraction(self, valuation_date, end_date) -> float:
        """Takes valuatio_date and end_date as strings, convert to datetime
            :: returns year_fraction (float) depending on the self.__convention"""
        #If dates arrives in the strings
        if type(valuation_date) == str:
            valuation_date = dt.strptime(valuation_date, self.__format_date)
        if type(end_date) == str:
            end_date = dt.strptime(end_date, self.__format_date)

        #We need to get the real "openned days" of the market (calendars) = Modified Following, etc.    
        if valuation_date.weekday()>=5 or valuation_date in self.__calendar:
            valuation_date = self.__apply_rolling_convention(valuation_date)
        if valuation_date.weekday()>=5 or valuation_date in self.__calendar:
            end_date = self.__apply_rolling_convention(end_date)
        return self.__convention_handler(valuation_date, end_date)

#Payment Schedule Handler:: first date of the schedule, last date of the schedule, periodicity of dates between, date format.
class PaymentScheduleHandler:
    """
    Classe pour générer des échéanciers de paiements entre une date de départ et une date de fin
    
    Inputs: 
        - valuation_date: date de départ (exemple: aujourd'hui ou t+2 = convention de marché)
        - end_date: dernière date de l'échéancier = date du dernier paiement
        - periodicity: temps entre deux paiements (monthly, quaterly, semi-annually, annually)
        - date_format: format d'input pour les valuation_date et end_date (exemple: %d/%m/%Y)

    utilisation: créer un payement scheduler avec les inputs, appeller build_schedule avec les conventions utilisées + marché.
    
    Renvoie un tuple d'échéances intermédiaires (ex:(0.5, 1, 1.5, 2, 2.5, 3) pour 3 ans paiement semi-annuel)
        ajusté aux jours de marchés fermés + convention de calculs.
    """
    def __init__(self, valuation_date: str, end_date: str, periodicity: str, date_format:str) -> None:
        self.__valuation_date = valuation_date
        self.__end_date = end_date
        self.__periodicity = periodicity
        self.__date_format = date_format
        pass

    def build_schedule(self, convention: str, rolling_convention: str, market: str) -> tuple:
        """Takes a start_date, end_date, periodicity 
            :: returns a tuple of year_fractions tuple because read only."""
        self.__valuation_date = dt.strptime(self.__valuation_date, self.__date_format)
        self.__end_date = dt.strptime(self.__end_date, self.__date_format)
        list_dates = self.__get_intermediary_dates()

        maturityhandler = Maturity_handler(convention, self.__date_format, rolling_convention, market)

        list_year_fractions = []
        #If we need the "t" corresponding to the first date/start date of the product (t=0), adjust list_dates[1:] to list_dates[0:]
        for date in list_dates[1:]:
            list_year_fractions.append(maturityhandler.get_year_fraction(list_dates[0], date))
        return list(list_year_fractions)

    def __get_intermediary_dates(self) -> list:
        """Build a dates list with all intermediary dates between start and end based on periodicity."""
        """Supported periodicity: monthly, quaterly, semi-annually, annually."""
        list_dates = [self.__valuation_date]
        count_date = self.__valuation_date
       
        while count_date < self.__end_date-relativedelta(months=1):
            if self.__periodicity == "monthly":
                count_date += relativedelta(months=1)
                list_dates.append(count_date)
            elif self.__periodicity == "quaterly":
                count_date += relativedelta(months=3)
                list_dates.append(count_date)
            elif self.__periodicity == "semi-annually":
                count_date += relativedelta(months=6)
                list_dates.append(count_date)
            elif self.__periodicity == "annually":
                count_date += relativedelta(years=1)
                list_dates.append(count_date)
            elif self.__periodicity == "none":
                count_date = self.__end_date
                list_dates.append(count_date)
            else:
                raise ValueError(f"Entered periodicity {self.__periodicity} is not supported. Supported periodicity: monthly, quaterly, semi-annually, annually, none.")
        
        list_dates.append(self.__end_date)
        return list_dates
        
#Classe de rate et courbe de taux
class Rates_curve:
    """
    Classe pour gérer les courbes de taux, interpoler, calculer les taux forward, shift de courbe, etc.

    Inputs:
        - path_rate: path du fichier csv contenant les taux
        - flat_rate: taux fixe à appliquer si besoin (optionnel, utilisé pour les courbes flats)

    Utilisation:
        - get_data_rate: retourne les données de la courbe de taux
        - year_fraction_data: ajoute une colonne Year_fraction à la dataframe des taux
        - attribute_rates_curve: ajoute les maturités des produits à la courbe de taux
        - linear_interpol: interpole les taux de la courbe de taux
        - quadratic_interpol: interpole les taux de la courbe de taux
        - Nelson_Siegel_interpol: interpole les taux de la courbe de taux
        - flat_rate: applique un taux fixe à la courbe de taux
        - forward_rate: calcule les taux forward de la courbe de taux
        - create_product_rate_curve: crée une courbe de taux pour un produit donné
        - shift_curve: shift la courbe de taux

    Renvoie la courbe de taux pour un produit donné.
    """
    def __init__(self, path_rate:str, flat_rate:float= None):
        self.__path_rate = path_rate
        self.__flat_rate = flat_rate

        self.__flat_rate = flat_rate
        self.__data_rate = pd.read_csv(path_rate,sep=";") #Only CSV reader supported for now. No link to external sources developped (not required + not pratical for student's dev).
        self.curve_rate_product = None
        pass

    def get_data_rate(self):
        """
        Fonction pour renvoyer les données de la courbe de taux (depuis un csv).
        """
        return self.__data_rate

    def year_fraction_data(self,convention: str=CONVENTION_DAY_COUNT) -> pd.DataFrame:
        """
        Fonction pour ajouter une colonne Year_fraction à la dataframe des taux.

        Input:
        - convention: convention de calcul de la year_fraction (30/360, 30/365, etc)
        """
        factor_map = {'D': 1, 'W': 7, 'M': 30, 'Y': convention}
        self.__data_rate['Year_fraction'] = self.__data_rate['Pillar'].str[:-1].astype(float) * self.__data_rate['Pillar'].str[-1].map(factor_map) / convention
        self.__data_rate['Year_fraction'] = self.__data_rate['Year_fraction'].round(6)
        return self.__data_rate

    def attribute_rates_curve(self,product_year_fraction: list) -> pd.DataFrame:
        """
        Fonction pour ajouter les maturités des produits à la courbe de taux.

        Input:
        - product_year_fraction: liste des maturités des produits
        """
        df= pd.DataFrame({"Year_fraction": product_year_fraction})
        df["Year_fraction"]=df["Year_fraction"].round(6)
        df = df[~df["Year_fraction"].isin(self.year_fraction_data(360)["Year_fraction"])]
        self.__data_rate = pd.merge(self.year_fraction_data(360),df,how='outer')
        self.__data_rate = self.__data_rate.sort_values(by='Year_fraction').reset_index(drop=True)
        return self.__data_rate

    def linear_interpol(self,product_year_fraction: list) -> pd.DataFrame:
        """
        Fonction pour interpoler les taux de la courbe de taux -> méthode linéaire.
        """
        self.__data_rate = self.attribute_rates_curve(product_year_fraction)
        self.__data_rate["Rate"] = self.__data_rate["Rate"].interpolate(method='linear')
        return self.__data_rate
    
    def quadratic_interpol(self,product_year_fraction: list) -> pd.DataFrame:
        """
        Fonction pour interpoler les taux de la courbe de taux -> méthode quadratic.
        """
        self.__data_rate = self.attribute_rates_curve(product_year_fraction)
        self.__data_rate["Rate"] = self.__data_rate["Rate"].interpolate(method='quadratic')
        return self.__data_rate

    def Nelson_Siegel_interpol(self,convention,product_year_fraction: list) -> pd.DataFrame:
        """
        Fonction pour interpoler les taux de la courbe de taux -> méthode Nelson-Siegel.
        """
        self.__data_rate = self.year_fraction_data(convention)
        Nelson_param = optimize_nelson_siegel(self.__data_rate["Year_fraction"],self.__data_rate["Rate"])
        self.__data_rate = self.attribute_rates_curve(product_year_fraction)
        for rates in self.__data_rate["Year_fraction"]:
            if self.__data_rate.loc[self.__data_rate["Year_fraction"]==rates,"Rate"].isna().any():
                self.__data_rate.loc[self.__data_rate["Year_fraction"]==rates,"Rate"] = nelson_siegel(rates, Nelson_param[0], Nelson_param[1], Nelson_param[2], Nelson_param[3])
        return self.__data_rate

    def flat_rate(self,product_year_fraction: list) -> pd.DataFrame:
        """
        Fonction pour construire une courbe un taux fixe.
        """
        self.__data_rate = self.attribute_rates_curve(product_year_fraction)
        self.__data_rate["Rate"] = self.__flat_rate
        return self.__data_rate

    def forward_rate(self,product_year_fraction:list, type_interpol:str=TYPE_INTERPOL) -> pd.DataFrame:
        """
        Fonction pour calculer les taux forward de la courbe de taux.

        Input:
        - product_year_fraction: liste des maturités des produits
        - type_interpol: type d'interpolation à utiliser pour la courbe de taux (Linear, Quadratic, Nelson_Siegel, Flat)
        """
        if type_interpol == "Linear":
            self.__data_rate = self.linear_interpol(product_year_fraction)
        if type_interpol == "Quadratic":
            self.__data_rate = self.quadratic_interpol(product_year_fraction)
        if type_interpol == "Nelson_Siegel":
            self.__data_rate = self.Nelson_Siegel_interpol(360,product_year_fraction)
        if type_interpol == "Flat":
            self.__data_rate = self.flat_rate(product_year_fraction)
        for i in range(len(self.__data_rate) - 1):
            year_fraction = self.__data_rate["Year_fraction"].iloc[i]
            next_year_fraction = self.__data_rate["Year_fraction"].iloc[i + 1]
            rate = self.__data_rate["Rate"].iloc[i]
            next_rate = self.__data_rate["Rate"].iloc[i + 1]
            self.__data_rate.at[i+1, "Forward_rate"] = ((((1 + next_rate) ** next_year_fraction) / ((1 + rate) ** year_fraction)) ** (1 / (next_year_fraction - year_fraction))) - 1
        return self.__data_rate
    
    def create_product_rate_curve(self,product_year_fraction: list, type_interpol:str=TYPE_INTERPOL) -> pd.DataFrame:
        """
        Fonction pour créer une courbe de taux pour un produit donné.

        Input:
        - product_year_fraction: liste des maturités des produits
        - type_interpol: type d'interpolation à utiliser pour la courbe de taux (Linear, Quadratic, Nelson_Siegel, Flat)
        """
        self.__data_rate = self.forward_rate(product_year_fraction,type_interpol)
        self.__data_rate = self.__data_rate[self.__data_rate["Year_fraction"].isin(product_year_fraction)]

        return self.__data_rate
    
    def shift_curve(self, shift:dict, type_interpol:str=TYPE_INTERPOL):
        """
        Fonction pour shifter la courbe des taux, possibilité d'utiliser un shift linéaire ou non.
        Input:
        - shift: dictionnaire avec les clés = maturité des produits à shifter, valeurs = shift à appliquer pour chaque maturités
        - type_interpol: type d'interpolation à utiliser pour la courbe de taux (Linear, Quadratic, Nelson_Siegel, Flat)
        """
        product_year_fraction = shift.keys()
        self.__data_rate = self.create_product_rate_curve(product_year_fraction,type_interpol)
        print(type(self.curve_rate_product))
        self.__data_rate['Rate']+=self.__data_rate['Year_fraction'].map(shift)
        self.__data_rate = self.create_product_rate_curve(product_year_fraction,type_interpol)
        pass

    def deep_copy(self,flat_rate:float=None):
        return Rates_curve(self.__path_rate,flat_rate)
    
    def change_rate(self,product_year_fraction: list, fixed_rate, type_interpol = TYPE_INTERPOL) -> pd.DataFrame:
        self.__data_rate = self.create_product_rate_curve(product_year_fraction,type_interpol)
        self.__data_rate['Rate']= fixed_rate
        return self.__data_rate
    
    def return_data_frame(self):
        return self.__data_rate

#Classe de vol

#Rates diffusion models: Vasicek, CIR, Hull-White(1F), HJM, Libor

#Price diffision models: Mouvement Brownien, (Jump Diffusion, Diffusion Stochastique what are those ?) + besoin de prise en compte des divs

#Helper to get the market from the currency.
def get_market(currency):
    if currency == "EUR":
        return "XECB"
    elif currency == "USD":
        return "XNYS"
    elif currency == "GBP":
        return "IFEU"
    elif currency == "BRL":
        return "BVMF"
    else:
        raise ValueError(f"Currency {currency} is not supported ! Choose: EUR, USD, GBP, BRL")

#Helper to calculate the yield of a fixed-income product.
def calculate_yield(cashflows: dict, market_price:float, initial_rate:float=INITIAL_RATE):
    """
    Solve for Yield to Maturity (YTM) given a dictionary of cashflows.
    
    :param cashflows: Dictionary where keys are time in years and values are cashflows.
    :param initial_rate: Initial guess for YTM.
    :return: Yield to Maturity (YTM)
    """
    def ytm(y):
        return sum([cf["NPV"] / (1 + y) ** t for t, cf in cashflows.items()]) - market_price
    
    ytm_solution = fsolve(ytm, initial_rate)[0]
    return ytm_solution*100
