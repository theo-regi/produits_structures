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
        'disp': False         # optional: shows progress in console
    }

#Delta to place each strikes on the local vol surface.
BASE_DELTA_K = 0.025 #think delta k as a percentage

#Base limits for strikes ie: -/+30% of the spot:
BASE_LIMITS_K = (0.7, 1.3) #50% of strike and 150% du strike = (0.5,1.5)

#Base method for heston calibration:
BASE_CALIBRATION_HESTON="Market" #Supported methods: "SSVI", "Market" / Market method is recommanded (faster calibration + reflects more the Volatility term structure of the market)

#Base maximum time to maturity to calibrate Heston model:
BASE_MAX_T= 5.0

#Base time interval between pricing_date and T:
BASE_T_INTERVAL=0.25

#Initial guess for Heston model:
INITIAL_HESTON= [0.1, 0.1, 0.1, 0.1, 0] #[0.05, 0.2, 0.2, 0.2, -0.5]  #v0, kappa, theta, eta, rho

#Base method for heston model calibration solver:
HESTON_METHOD='L-BFGS-B' #Supported methods: 'L-BFGS-B', 'SLSQP', but you will prefer L-BFGS-B for those basic problems

#Bounds for heston parameters:
HESTON_BOUNDS=(
    (1e-4, 10.0), #v0
    (1e-4, 5.0), #kappa
    (1e-4, 5.0), #theta
    (1e-4, 5.0), #eta
    (-0.9999,0.9999)  #rho
)

#Options for Heston model calibration:
HESTON_CALIBRATION_OPTIONS={
       'ftol': 1e-6,       # tolerance for convergence
        'maxiter': 100,    # iteration limit
        'disp': True         # optional: shows progress in console
}

#Base limits of moneyness for heston (reduced to accelerate the calibration):
if BASE_CALIBRATION_HESTON=='Market': BASE_LIMITS_K_H= (0.07, 1.30)
else: BASE_LIMITS_K_H=(0.085, 1.15) #85% of strike and 115% du strike = (0.85,1.15), we do recommand larger moneyness calibration

#Fraction of the options to retain for calibration.
if BASE_CALIBRATION_HESTON=='Market': CUTOFF_H=1 #80% of the options will be used for calibration
else: CUTOFF_H=0.5

#Number of cores to use for parallelization:
N_CORES=1

#__________________________Heston simulation CONSTANTS:_______________________
#Number of paths for the simulation
NUMBER_PATHS_H = 100

#Number of steps for the simulation
NB_STEPS_H=100

#_______________________________ENUM CONSTANTS:_______________________________
#Enum for the type of options: CALL or PUT
class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0