from constants import OptionType, BASE_SPOT, BASE_STRIKE, BASE_RATE, CONVENTION_DAY_COUNT, ROLLING_CONVENTION, BASE_NOTIONAL, FORMAT_DATE, BASE_CURRENCY, BASE_MODEL, BASE_SIGMA, BASE_DIV_RATE
from products import VanillaOption
from models import BSM
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
    def __init__(self, start_date:str, end_date:str, type:str=OptionType.CALL, model:str=BASE_MODEL, spot:float=BASE_SPOT, strike:float=BASE_STRIKE, div_rate:float=BASE_DIV_RATE, day_count:str=CONVENTION_DAY_COUNT, rolling_conv:str=ROLLING_CONVENTION, notional:float=BASE_NOTIONAL, format_date:str=FORMAT_DATE, currency:str=BASE_CURRENCY, sigma:float=None, rate:float=BASE_RATE) -> None:
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

    @property
    def price(self):
        if self._model == "Black-Scholes-Merton":
            if self._sigma is None:
                self._sigma = BASE_SIGMA
            
            self._option = VanillaOption(start_date=self._start_date, end_date=self._end_date, type=self._type, strike=self._strike, notional=self._notional, currency=self._currency, div_rate=self._div_rate)
            self._model = dict_models[self._model](self._sigma, self._option)
            return self._model.price(self._spot)
        
    @property
    def delta(self):
        if self._model == "Black-Scholes-Merton":
            if self._sigma is None:
                self._sigma = BASE_SIGMA
            
            self._option = VanillaOption(start_date=self._start_date, end_date=self._end_date, type=self._type, strike=self._strike, notional=self._notional, currency=self._currency, div_rate=self._div_rate)
            self._model = dict_models[self._model](self._sigma, self._option)
            return self._model.delta(self._spot)

    @property
    def gamma(self):
        if self._model == "Black-Scholes-Merton":
            if self._sigma is None:
                self._sigma = BASE_SIGMA
            
            self._option = VanillaOption(start_date=self._start_date, end_date=self._end_date, type=self._type, strike=self._strike, notional=self._notional, currency=self._currency, div_rate=self._div_rate)
            self._model = dict_models[self._model](self._sigma, self._option)
            return self._model.gamma(self._spot)

    @property
    def vega(self):
        if self._model == "Black-Scholes-Merton":
            if self._sigma is None:
                self._sigma = BASE_SIGMA
            
            self._option = VanillaOption(start_date=self._start_date, end_date=self._end_date, type=self._type, strike=self._strike, notional=self._notional, currency=self._currency, div_rate=self._div_rate)
            self._model = dict_models[self._model](self._sigma, self._option)
            return self._model.vega(self._spot)

    @property
    def theta(self):
        if self._model == "Black-Scholes-Merton":
            if self._sigma is None:
                self._sigma = BASE_SIGMA
            
            self._option = VanillaOption(start_date=self._start_date, end_date=self._end_date, type=self._type, strike=self._strike, notional=self._notional, currency=self._currency, div_rate=self._div_rate)
            self._model = dict_models[self._model](self._sigma, self._option)
            return self._model.theta(self._spot, self._rate)

    @property
    def rho(self):
        if self._model == "Black-Scholes-Merton":
            if self._sigma is None:
                self._sigma = BASE_SIGMA
            
            self._option = VanillaOption(start_date=self._start_date, end_date=self._end_date, type=self._type, strike=self._strike, notional=self._notional, currency=self._currency, div_rate=self._div_rate)
            self._model = dict_models[self._model](self._sigma, self._option)
            return self._model.rho(self._spot, self._rate)

