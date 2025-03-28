import enum
#-------------------------------------------------------------------------------------------------------
#----------------------------Script pour modifier les constantes de base de l'appli---------------------
#-------------------------------------------------------------------------------------------------------
#GENERAL UTILITIES CONSTANTS:
#Base type of interpolation for curves, will be used if not provided by the user.
TYPE_INTERPOL = 'Quadratic' #Supported types: 'Linear', 'Quadratic', 'Nelson_Siegel', 'Flat'

#Base format for dates, will be used if not provided by the user.
FORMAT_DATE = '%d/%m/%Y'

#Default convention for day_count, will be used if not provided by the user.
CONVENTION_DAY_COUNT = '30/360' #Supported conventions: '30/360', 'ACT/360', 'ACT/365', 'Act/Act'

#Default convention for rolling on closed days, will be used if not provided by the user.
ROLLING_CONVENTION = 'Modified Following' #Supported conventions: 'Following', 'Modified Following', 'Preceding', 'Modified Preceding'

#Solver methode for optimization, will be used if not provided by the user.
SOLVER_METHOD = 'L-BFGS-B' #Supported methods: 'L-BFGS-B', 'SLSQP', 'Powell', 'TNC'

#GENERAL FINANCIAL PRODUCTS CONSTANTS:
#Base notional for the instruments, will be used if not provided by the user.
BASE_NOTIONAL = 100 #Use 100 for percentage

#FIXED INCOME PRODUCTS CONSTANTS:
#Base shift for the instruments, will be used if not provided by the user.
BASE_SHIFT = 0.01   #Use 0.01 for 1bps

#Exchange notional: True or False (True for Bonds / False for Swap Legs)
EXCHANGE_NOTIONAL = False

#For Yield calculation, initial guess for the solver
INITIAL_RATE = 0.05

#EQD PRODUCTS CONSTANTS:
#Base Spot for EQD pricings, will be used if not provided by the functions.
BASE_SPOT = 100

#Base Strike for EQD pricings, will be used if not provided by the functions. = ATM
BASE_STRIKE = 100

#Base rate for EQD pricings, will be used if discounting rates is not provided.
BASE_RATE=0.05

#Base dividend rate for EQD pricings, will be used if not provided by the functions.
BASE_DIV_RATE=0.03

#Base currency for EQD pricings, will be used if not provided by the functions.
BASE_CURRENCY='EUR'

#Base Model for options valuation
BASE_MODEL = 'Black-Scholes-Merton'

#Base sigma for options (fixed implied vol)
BASE_SIGMA = 0.05

#Enum for the type of options: CALL or PUT
class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0