#FRED API key : 1ff5cd3458d773c477a0ff3ce230c823 
from fredapi import Fred
import pandas as pd
import os

class YieldCurveFetcher:
    def __init__(self, api_key):
        self.fred = Fred(api_key=api_key)
        # Mapping of maturities to FRED series IDs (update as needed)
        self.series_map = {
            "3M": "DGS3MO",
            "6M": "DGS6MO",
            "1Y": "DGS1",
            "2Y": "DGS2",
            "3Y": "DGS3",
            "5Y": "DGS5",
            "7Y": "DGS7",
            "10Y": "DGS10",
            "20Y": "DGS20",
            "30Y": "DGS30"
        }

    def get_rate_history(self, pillar, start_date, end_date):
        """Fetches historical rate series for a given pillar."""
        series_id = self.series_map.get(pillar)
        if not series_id:
            raise ValueError(f"No FRED series mapped for pillar {pillar}")
        data = self.fred.get_series(series_id, start_date=start_date, end_date=end_date)
        return data

    def get_full_yield_curve_history(self, start_date, end_date):
        """Fetches historical data for all defined maturities."""
        df = pd.DataFrame()
        for pillar, series_id in self.series_map.items():
            df[pillar] = self.fred.get_series(series_id, start_date=start_date, end_date=end_date)
        return df

if __name__ == "__main__":
    api_key = "1ff5cd3458d773c477a0ff3ce230c823"
    fetcher = YieldCurveFetcher(api_key)
    path = os.path.dirname(os.path.abspath(__file__))
    start_date = "2023-03-26"
    end_date = "2025-03-26"
    # Get full curve history
    curve_df = fetcher.get_full_yield_curve_history(start_date, end_date)
    curve_df.to_csv(path + "/yield_curve_history.csv", index=True)
