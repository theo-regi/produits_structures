import numpy as np

import unittest
from products import ZCBond

class TestZCBond(unittest.TestCase):
    def setUp(self):
        """Initialize a ZCBond with a default nominal value"""
        self.bond = ZCBond()

    def test_npv_from_df(self):
        """Test NPV calculation using a discount factor (0.95) nominal is not given (=100)"""
        self.assertEqual(self.bond.get_npv_zc_from_df(0.95), 95)

    def test_npv_from_zcrate(self):
        """Test NPV calculation using rate and maturity"""
        rate, maturity = 0.05, 5
        expected_npv = 100 * (np.exp(-rate * maturity))
        self.assertEqual(self.bond.get_npv_zc_from_zcrate(rate, maturity), expected_npv)

    def test_discount_factor_from_zcrate(self):
        """Test discount factor calculation"""
        rate, maturity = 0.04, 3
        expected_df = np.exp(-rate * maturity)
        self.assertEqual(self.bond.get_discount_factor_from_zcrate(rate, maturity), expected_df)

    def test_get_zc_rate(self):
        """Test Zero-Coupon rate calculation from discount factor"""
        discount_factor, maturity = 0.90, 4
        expected_rate = -np.log(discount_factor) / maturity
        self.assertEqual(self.bond.get_zc_rate(discount_factor, maturity), expected_rate)

    def test_get_ytm(self):
        """Test Yield to Maturity calculation"""
        market_price, maturity = 80, 5
        expected_ytm = (100 / market_price) ** (1 / maturity) - 1
        self.assertEqual(self.bond.get_ytm(market_price, maturity), expected_ytm)

    def test_get_duration_macaulay(self):
        """Test Macaulay duration (should return maturity)"""
        maturity = 7
        self.assertEqual(self.bond.get_duration_macaulay(maturity), maturity)

    def test_get_modified_duration(self):
        """Test Modified Duration calculation"""
        market_price, maturity = 85, 4
        ytm = (100 / market_price) ** (1 / maturity) - 1
        expected_mod_duration = maturity / (1 + ytm)
        self.assertEqual(self.bond.get_modified_duration(market_price, maturity), expected_mod_duration)

    def test_get_sensitivity(self):
        """Test Sensitivity calculation"""
        new_rate, maturity = 0.03, 6
        expected_sensitivity = maturity / (1 + new_rate)
        self.assertEqual(self.bond.get_sensitivity(new_rate, maturity), expected_sensitivity)

    def test_get_convexity_issued(self):
        """Test Convexity of an issued Zero-Coupon Bond"""
        maturity, market_price = 5, 80
        ytm = (100 / market_price) ** (1 / maturity) - 1
        expected_convexity = (maturity * (maturity + 1) * 100) / (market_price * ((1 + ytm) ** (maturity + 2)))
        self.assertEqual(self.bond.get_convexity(maturity, market_price=market_price), expected_convexity)

    def test_get_convexity_not_issued(self):
        """Test Convexity of a non-issued Zero-Coupon Bond"""
        maturity, discount_factor = 5, 0.85
        market_price = 85
        ytm = (100 / market_price) ** (1 / maturity) - 1
        expected_convexity = (maturity * (maturity + 1) * 100) / (market_price * ((1 + ytm) ** (maturity + 2)))
        self.assertEqual(self.bond.get_convexity(maturity, discount_factor=discount_factor), expected_convexity)

    def test_invalid_convexity_inputs(self):
        """Test error handling for incorrect convexity inputs"""
        maturity = 5
        with self.assertRaises(ValueError):
            self.bond.get_convexity(maturity)

    def test_invalid_ytm_maturity(self):
        """Test YTM with zero maturity should raise an error"""
        with self.assertRaises(ValueError):
            self.bond.get_ytm(90, 0)

if __name__ == "__main__":
    unittest.main()
