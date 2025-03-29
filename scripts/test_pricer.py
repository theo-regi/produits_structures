import unittest
from pricers import OptionPricer
from constants import OptionType

class TestPricer(unittest.TestCase):
    def setUp(self):
        self.pricer =OptionPricer(
            start_date="28/03/2025",
            end_date="28/03/2026",
            type=OptionType.CALL,
            model="Black-Scholes-Merton",
            spot=100,
            strike=105,
            div_rate=0.03,
            currency="EUR",
            sigma=0.08,
            rate=0.05,
            notional=1)

    def test_price(self):
        # Example test case for price calculation
        self.assertAlmostEqual(self.pricer.price, 1.925485846, places=4)

    def test_delta(self):
        # Example test case for delta calculation
        self.assertAlmostEqual(self.pricer.delta, 0.36346171, places=4)

    def test_gamma(self):
        # Example test case for gamma calculation
        self.assertAlmostEqual(self.pricer.gamma, 0.045980369, places=4)

    def test_vega(self):
        # Example test case for vega calculation
        self.assertAlmostEqual(self.pricer.vega, 36.78429503, places=4)

    def test_theta(self):
        # Example test case for theta calculation
        self.assertAlmostEqual(self.pricer.theta, -2.465482639, places=6)

    def test_rho(self):
        # Example test case for rho calculation
        self.assertAlmostEqual(self.pricer.rho, 34.42068517, places=4)

    def test_payoff(self):
        # Example test case for payoff calculation
        self.pricer.price
        self.assertAlmostEqual(self.pricer.payoff, 2.024207616, places=4)

if __name__ == "__main__":
    unittest.main()
