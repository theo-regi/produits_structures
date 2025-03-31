from constants import OptionType, BASE_SPOT, BASE_STRIKE, BASE_RATE, CONVENTION_DAY_COUNT, ROLLING_CONVENTION, BASE_NOTIONAL,\
    FORMAT_DATE, BASE_CURRENCY, BASE_MODEL, BASE_SIGMA, BASE_DIV_RATE, BASE_METHOD_VOL, TOLERANCE, MAX_ITER, BOUNDS, STARTING_POINT
from products import VanillaOption
from utils import ImpliedVolatilityFinder, SVIParamsFinder
from models import BSM
import numpy as np
dict_models = {"Black-Scholes-Merton": BSM}
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
            #print(self.payoff)
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
            return svi_params_finder.find_svi_parameters()
        else:
            raise ValueError("Didn't find SVI Parameters with these inputs!")