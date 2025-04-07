import enum
#-------------------------------------------------------------------------------------------------------
#----------------------------Script pour modifier les constantes de base de l'appli---------------------
#-------------------------------------------------------------------------------------------------------
#_______________________________GENERAL UTILITIES CONSTANTS:_______________________________
#Base type of interpolation for curves, will be used if not provided by the user.
TYPE_INTERPOL = 'Quadratic' #Supported types: 'Linear', 'Quadratic', 'Nelson_Siegel', 'Flat'

#Base format for dates, will be used if not provided by the user.
FORMAT_DATE = '%d/%m/%Y'

#Default convention for day_count, will be used if not provided by the user.
CONVENTION_DAY_COUNT = '30/360' #Supported conventions: '30/360', 'ACT/360', 'ACT/365', 'Act/Act'

#Default convention for rolling on closed days, will be used if not provided by the user.
ROLLING_CONVENTION = 'Modified Following' #Supported conventions: 'Following', 'Modified Following', 'Preceding', 'Modified Preceding'

#Solver method for optimization, will be used for basic solvers (Yield/Implied vol/rates) if not provided by the user.
SOLVER_METHOD = 'L-BFGS-B' #Supported methods: 'L-BFGS-B', 'SLSQP', 'Powell', 'TNC', but you will prefer L-BFGS-B for those basic problems

#Solver method for the SVI parameters optimization, will be used in the ImpledVolatilityFinder class in the SVI_params function:
SVI_SOLVER_METHOD = 'SLSQP' #Prefered method will be SLSQP because of all the constraints and bounds we have to set for the SVI parameters optimization.

#SSVI solver method:
SSVI_METHOD = 'SLSQP'

#Initial Parameters Nielson-Siegel:
INITIAL_NS = [1,1,1,1]

#Tolerance for the solver
TOLERANCE = 1e-8

#Max iterations for the solver
MAX_ITER = 1000

#Bounds for the solver (volatility)
BOUNDS = (1e-4, 5.0)

#Starting point for the solver (volatility)
STARTING_POINT = 0.2

#_______________________________GENERAL FINANCIAL PRODUCTS CONSTANTS:_______________________________
#Base notional for the instruments, will be used if not provided by the user.
BASE_NOTIONAL = 100 #Use 100 for percentage

#_______________________________FIXED INCOME PRODUCTS CONSTANTS:_______________________________
#Base shift for the instruments, will be used if not provided by the user.
BASE_SHIFT = 0.01   #Use 0.01 for 1bps

#Exchange notional: True or False (True for Bonds / False for Swap Legs)
EXCHANGE_NOTIONAL = False

#For Yield calculation, initial guess for the solver
INITIAL_RATE = 0.05

#_______________________________EQD PRODUCTS CONSTANTS:_______________________________
#Base Spot for EQD pricings, will be used if not provided by the functions.
BASE_SPOT = 100

#Base Strike for EQD pricings, will be used if not provided by the functions. = ATM
BASE_STRIKE = 100

#Base rate for EQD pricings, will be used if discounting rates is not provided.
BASE_RATE=0.05

#Base dividend rate for EQD pricings, will be used if not provided by the functions.
BASE_DIV_RATE=0

#Base currency for EQD pricings, will be used if not provided by the functions.
BASE_CURRENCY='EUR'

#Base Model for options valuation
BASE_MODEL = 'Black-Scholes-Merton'

#Base sigma for options (fixed implied vol)
BASE_SIGMA = 0.05

#______________________________IMPLIED VOLATILITY CALCULATION CONSTANTS:_______________________________
BASE_METHOD_VOL = 'Dichotomy' #Method used for implied volatility calculation (for utils.volatility): Dichotomy, Optimization, Newton-Raphson

#Supported methods: 'Dichotomy', 'Optimization', 'Newton-Raphson'
IMPLIED_VOL_METHODS = {     #Used to directly map to the supported properties
    "Dichotomy": "_dichotomy",
    "Optimization": "_optimization",
    "Newton-Raphson": "_newton_raphson"
}

#Initial guess for the SVI parameters (a,b,p,m,sigma)
INITIAL_SVI = [0.1, 0.1, 0.1, 0.1, 0.1]

#Solver parameters
OPTIONS_SOLVER_SVI = {
        'ftol': 1e-6,       # tolerance for convergence
        'maxiter': 200,    # iteration limit
        'disp': False         # optional: shows progress in console
    }

#Moneyness bounds: to take only slight OTM/ATM options.
BOUNDS_MONEYNESS = (0.7, 1.3)   #Avoid using to much over OTM / ATM options for vol calibrations

#Calibrate vol on only OTM options (recommanded because liquier).
OTM_CALIBRATION = True #True for OTM calibration, False to take OTM and ATM options for given bounds calibration

#Calibrate on volume (will use volumes over x% of the average volume)
VOLUME_CALIBRATION = True

#Threshold for volume calibration (under average):
VOLUME_THRESHOLD = 0.7

#SSVI initial guess for the parameters (k, v_o, b_inf, p, mu, l):
INITIAL_SSVI = [0.5, 0.04, 0.01, 0.1, 0.1, 0.1]

#Solver parameters
OPTIONS_SOLVER_SSVI = {
        'ftol': 1e-6,       # tolerance for convergence
        'maxiter': 100,    # iteration limit
        'disp': True         # optional: shows progress in console
    }

#_______________________________ENUM CONSTANTS:_______________________________
#Enum for the type of options: CALL or PUT
class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0