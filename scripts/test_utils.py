import unittest
from utils import Maturity_handler, PaymentScheduleHandler, Rates_curve
from constants import OptionType
from products import OptionMarket, OptionPricer

#-------------------------------------------------------------------------------------------------------
#----------------------------Script pour tester les classes unitaires-----------------------------------
#-------------------------------------------------------------------------------------------------------
class TestMaturityHandlerDayCount(unittest.TestCase):
    def setUp(self):
        """Initialize Maturity_handler instances for different conventions"""
        self.date_format = "%d/%m/%Y"
        self.start_date = "02/01/2024"
        self.end_date = "1/07/2025"

        self.maturity_30_360 = Maturity_handler("30/360", self.date_format, "Following", "XECB")
        self.maturity_act_360 = Maturity_handler("Act/360", self.date_format, "Following", "XECB")
        self.maturity_act_365 = Maturity_handler("Act/365", self.date_format, "Following", "XECB")
        self.maturity_act_act = Maturity_handler("Act/Act", self.date_format, "Following", "XECB")

    def test_30_360(self):
        """Test 30/360 convention"""
        expected_result = 1.497222
        result = self.maturity_30_360.get_year_fraction(self.start_date, self.end_date)
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_act_360(self):
        """Test Act/360 convention"""
        expected_result = 1.516667
        result = self.maturity_act_360.get_year_fraction(self.start_date, self.end_date)
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_act_365(self):
        """Test Act/365 convention"""
        expected_result = 1.495890
        result = self.maturity_act_365.get_year_fraction(self.start_date, self.end_date)
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_act_act(self):
        """Test Act/Act convention"""
        expected_result = 1.490426
        result = self.maturity_act_act.get_year_fraction(self.start_date, self.end_date)
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_invalid_convention(self):
        """Test invalid convention handling"""
        maturity_invalid = Maturity_handler("Invalid", self.date_format, "Following", "XECB")
        with self.assertRaises(ValueError):
            maturity_invalid.get_year_fraction(self.start_date, self.end_date)

    def test_month_end_adjustment_30_360(self):
        """Test 30/360 convention when start or end date falls on the 31st"""
        maturity_30_360 = Maturity_handler("30/360", self.date_format, "Following", "XECB")
        start_date = "31/01/2024"
        end_date = "28/02/2024"

        expected_result = 0.077778
        result = maturity_30_360.get_year_fraction(start_date, end_date)
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_following_convention(self):
        """Test Following rolling convention"""
        maturity_following = Maturity_handler("Act/360", self.date_format, "Following", "XECB")
        start_date = "01/01/2025"
        end_date = "02/01/2025"
        result = maturity_following.get_year_fraction(start_date, end_date)
        self.assertEqual(result, 0)

    def test_modified_following_convention(self):
        """Test Modified Following rolling convention"""
        maturity_modified_following = Maturity_handler("Act/360", self.date_format, "Modified Following", "XECB")
        start_date = "31/03/2025"
        end_date = "01/04/2025"
        result = maturity_modified_following.get_year_fraction(start_date, end_date)
        self.assertGreater(result, 0)

    def test_preceding_convention(self):
        """Test Preceding rolling convention"""
        maturity_preceding = Maturity_handler("Act/360", self.date_format, "Preceding", "XECB")
        start_date = "01/01/2025"
        end_date = "31/12/2024"
        result = maturity_preceding.get_year_fraction(start_date, end_date)
        self.assertEqual(result, 0)

    def test_modified_preceding_convention(self):
        """Test Modified Preceding rolling convention + Test on a week-end day"""
        maturity_modified_preceding = Maturity_handler("Act/360", self.date_format, "Modified Preceding", "XECB")
        start_date = "01/03/2025"
        end_date = "03/03/2025"
        result = maturity_modified_preceding.get_year_fraction(start_date, end_date)
        self.assertEqual(result, 0)

class TestPaymentScheduleHandler(unittest.TestCase):
    def setUp(self):
        """Set up test cases with different periodicities."""
        self.date_format = "%d/%m/%Y"
        self.valuation_date = "02/01/2024"
        self.end_date = "02/01/2026"

    def test_length(self):
        """Test the length matching in the schedule"""
        schedule_handler = PaymentScheduleHandler(self.valuation_date, self.end_date, "monthly", self.date_format)
        result = schedule_handler.build_schedule("30/360", "Modified Following", "XECB")
        self.assertEqual(len(result), 24)

    def test_monthly_schedule(self):
        """Test monthly periodicity."""
        schedule_handler = PaymentScheduleHandler(self.valuation_date, self.end_date, "monthly", self.date_format)
        result = schedule_handler.build_schedule("30/360", "Modified Following", "XECB")
        expected_tuple = (0.0833,0.1667,0.25,0.3333,0.4167,0.5,0.5833,0.6667,0.75,0.8333,0.9167,1,1.0833,1.1667,1.25,1.3333,1.4167,1.5,1.5833,1.6667,1.75,1.8333,1.9167,2)
        for expected, actual in zip(result, expected_tuple):
            self.assertAlmostEqual(expected, actual, places=4)

    def test_quarterly_schedule(self):
        """Test quarterly periodicity."""
        schedule_handler = PaymentScheduleHandler(self.valuation_date, self.end_date, "quaterly", self.date_format)
        result = schedule_handler.build_schedule("30/360", "Modified Following", "XECB")
        expected_tuple = (0.25,0.5,0.75,1,1.25,1.5,1.75,2)
        for expected, actual in zip(result, expected_tuple):
            self.assertAlmostEqual(expected, actual, places=4)

    def test_semi_annual_schedule(self):
        """Test semi-annual periodicity."""
        schedule_handler = PaymentScheduleHandler(self.valuation_date, self.end_date, "semi-annually", self.date_format)
        result = schedule_handler.build_schedule("30/360", "Modified Following", "XECB")
        expected_tuple = (0.5,1,1.5,2)
        for expected, actual in zip(result, expected_tuple):
            self.assertAlmostEqual(expected, actual, places=4)

    def test_annual_schedule(self):
        """Test annual periodicity."""
        schedule_handler = PaymentScheduleHandler(self.valuation_date, self.end_date, "annually", self.date_format)
        result = schedule_handler.build_schedule("30/360", "Modified Following", "XECB")
        expected_tuple = (1,2)
        for expected, actual in zip(result, expected_tuple):
            self.assertAlmostEqual(expected, actual, places=4)

    def test_unsupported_periodicity(self):
        """Test invalid periodicity handling."""
        schedule_handler = PaymentScheduleHandler(self.valuation_date, self.end_date, "bi-monthly", self.date_format)
        with self.assertRaises(ValueError):
            schedule_handler.build_schedule("30/360", "Following", "XECB")

class TestRatesCurve(unittest.TestCase):
    def setUp(self):
        """Set up test cases with different periodicities."""
        self.flat_rate = 0.025
        self.path_rate = "RateCurve.csv"
    
    def test_linear_interpol(self):
        curve = Rates_curve(self.path_rate, self.flat_rate)

        #On fait le test sur le taux 2 Mois pour voir si ça fonctionne 
        liste= [0.002778,0.019444444,0.083333333,0.25,0.166666666666667]

        expected_result = 2.359799
        result= curve.linear_interpol(liste).iloc[3]['Rate']
        self.assertAlmostEqual(result, expected_result,places = 6)

    def test_quadratic_interpol(self):
        curve = Rates_curve(self.path_rate, self.flat_rate)

        #On fait le test sur le taux 2 Mois pour voir si ça fonctionne 
        liste= [0.002778,0.019444444,0.083333333,0.25,0.166666666666667]

        expected_result = 2.364787
        result= curve.quadratic_interpol(liste).iloc[3]['Rate']
        self.assertAlmostEqual(result, expected_result,places = 6)

    def test_Nelson_Siegel(self):
        liste= [0.002778,0.019444444,0.083333333,0.25,0.166666666666667]
        curve = Rates_curve(self.path_rate, self.flat_rate)
        result =curve.Nelson_Siegel_interpol(360,liste)
        self.assertFalse(result['Rate'].isna().any())

    def test_flat_rate(self):
        curve = Rates_curve(self.path_rate, self.flat_rate)
        liste= [0.002778,0.019444444,0.083333333,0.25,0.166666666666667]
        result = curve.flat_rate(liste)
        self.assertEqual(result['Rate'].iloc[0],0.025)

    def test_forward_rate(self):
        curve = Rates_curve(self.path_rate, self.flat_rate)
        liste= [0.002778,0.019444444,0.083333333,0.25,0.166666666666667]
        result = curve.forward_rate(liste,'Quadratic')
        self.assertFalse(result['Forward_rate'][1:].isna().any())

    def test_create_product_rate_curve(self):
        curve = Rates_curve(self.path_rate, self.flat_rate)
        liste= [0.002778,0.019444444,0.083333333,0.25,0.166666666666667]
        liste = [round(x, 6) for x in liste] 
        result = curve.create_product_rate_curve(liste,'Quadratic')
        print(result)
        self.assertFalse(result['Rate'].isna().any())

    def test_shift_rate_curve(self):
        curve = Rates_curve(self.path_rate, self.flat_rate)

        liste = [0.002778,0.019444444,0.083333333,0.25,0.166666666666667]
        liste = [round(x, 6) for x in liste]
        list_shifts = [0.01, 0.02, 0.03, 0.04, 0.05]
        dict_shifts = dict(zip(liste, list_shifts))

        result = curve.create_product_rate_curve(liste,'Quadratic')
        print("Non Shifted curve:")
        print(result)

        shift_rate_curve = curve.deep_copy()
        shift_rate_curve.shift_curve(dict_shifts, 'Quadratic')
        shifted_rates = shift_rate_curve.get_data_rate()
        print("Shifted curve:")
        print(shifted_rates)

        col1 = result['Rate'].tolist()
        col2 = shifted_rates['Rate'].to_list()
        self.assertTrue(any(a != b for a, b in zip(col1, col2)))

        col1 = result['Forward_rate'].tolist()
        col2 = shifted_rates['Forward_rate'].to_list()
        self.assertTrue(any(a != b for a, b in zip(col1, col2)))

class TestImpliedVolatilityFinder(unittest.TestCase):
    def setUp(self):
        """Set up test cases with different parameters."""
        self.pricer =OptionPricer(
            start_date="12/03/2025",
            end_date="14/03/2025",
            type=OptionType.CALL,
            model="Black-Scholes-Merton",
            spot=216.98,
            strike=217.5,
            div_rate=0,
            currency="EUR",
            rate=0,
            notional=1,
            price=2.52)
    
    def test_implied_volatility_dichotomie(self):
        """Test implied volatility calculation using dichotomie method."""
        result = self.pricer.implied_vol(method="Dichotomy")
        self.assertAlmostEqual(result, 0.42363, places=2)

    def test_implied_volatility_optimization(self):
        result = self.pricer.implied_vol(method='Optimization')
        self.assertAlmostEqual(result, 0.42363, places=2)

    def test_implied_volatility_newton_raphson(self):
        result = self.pricer.implied_vol(method='Newton-Raphson')
        self.assertAlmostEqual(result, 0.42363, places=2)

class TestSVIParamsFinder(unittest.TestCase):
    def setUp(self):
        """Set up test cases with different parameters."""
        start, end = "12/03/2025", "16/05/2025"
        type=OptionType.CALL
        model="Black-Scholes-Merton"
        spot=216.98
        self.types = [OptionType.CALL]*42
        self.strikes = [100,110,115,130,135,140,145,150,155,160,165,170,175,180,185,190,195,200,205,210,215,220,225,230,235,240,245,250,255,260,265,270,275,280,285,290,295,300,305,310,315,320]
        self.prices=[123.5,107.35,102.7,91.64,114.19,78.58,104.42,68.2,71.8,57.77,58.27,49.86,47.13,40.65,34.49,30.7,27.65,23.55,20,16.35,13.5,10.85,8.35,6.21,4.6,3.3,2.34,1.63,1.13,0.81,0.58,0.43,0.32,0.23,0.18,0.15,0.11,0.11,0.1,0.08,0.04,0.05]

        self.pricer = OptionPricer(
            start_date=start,
            end_date=end,
            type=type,
            model=model,
            spot=spot,
            div_rate=0,
            currency="EUR",
            rate=0,
            notional=1)

    def test_svi_params(self):
        import numpy as np
        result = self.pricer.svi_params(self.types, self.strikes, self.prices)
        print(result)
        self.assertEqual(len(result), 5)
        self.assertEqual(all([np.isfinite(x) for x in result]), True)
        self.assertTrue(result[1] > 0)  #Check b
        self.assertTrue(abs(result[2]) < 1)  #Check rho
        self.assertTrue(result[4] > 0) #Check sigma
        self.assertTrue(result[0]+result[1]*result[4]*np.sqrt(1-result[2]**2)>=0)
        
class TestOptionMarket_SSVI_Connection(unittest.TestCase):
    def setUp(self):
        OTM_v = True
        self.option_market = OptionMarket("data/options.csv", "data/underlying_prices.csv")
        p_date = "13/03/2025"
        maturity = '21/03/2025' #Checking first pricing date, first maturity "16/05/2025"

        list_types, list_strikes, list_prices, spot, t_options = self.option_market.get_values_for_calibration_SVI(p_date, maturity,(0.7, 1.3) , OTM_v)
        self.pricer = OptionPricer(p_date, maturity, model="Black-Scholes-Merton", spot=spot, div_rate=0, currency="EUR", rate=0, notional=1)

        self.params = self.pricer.svi_params(list_types, list_strikes, list_prices)

    def test_connection(self):
        """Test the connection to the SSI server."""
        self.assertIsInstance(list(self.params), list)
        self.assertEqual(len(self.params), 5)
        print(self.params)
        pass