from unittest import TestCase
import unittest

from datetime import datetime as dt

from utils import Maturity_handler
from utils import PaymentScheduleHandler

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
        """Initialize PaymentScheduleHandler instances for different conventions"""
        self.valuation_date = "02/01/2025"
        self.end_date = "02/01/2055"
        self.date_format ="%d/%m/%Y"

        self.Monthly_Schedule = PaymentScheduleHandler(self.valuation_date,self.end_date,"monthly",self.date_format)

    def test_intermediary_monthly(self):
        """test get_intermediary_dates"""

        expected_result = "02/02/2025"
        result = self.Monthly_Schedule.get_intermediary_dates()
        self.assertEqual(result[1], expected_result)

if __name__ == "__main__":
    unittest.main()