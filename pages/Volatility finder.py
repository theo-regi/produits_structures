import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scripts.products import SSVICalibration, DupireLocalVol, OptionMarket
import constants
#-----------------------------------------------------------------------------------------------------
#------------------------------------Page construction du portefeuille---------------------------------
#-----------------------------------------------------------------------------------------------------
#______________________________________INITIALISATION PAGE____________________________________________
STYLE_PATH=st.session_state.STYLE_PATH
st.set_page_config(page_title="📈 Volatility Models Viewer", layout="wide")
def apply_css(STYLE_PATH):
    with open(STYLE_PATH, "r") as f:
        css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
apply_css(STYLE_PATH)

st.title("🧠 Volatility Surface Analysis")
# Use pricing date from session
pricing_date = st.session_state.pricing_date
st.info(f"📅 Pricing Date: **{pricing_date}**")

# Tabs for SVI, SSVI, Dupire
svi_tab, ssvi_tab, dupire_tab = st.tabs(["🔹 SVI", "🔸 SSVI", "🔻 Dupire Local Vol"])

with svi_tab:
    st.header("SVI Volatility Slice")
    cache_key = (constants.FILE_PATH, constants.FILE_UNDERLYING)
    svis = constants.get_from_cache("SVI_PARAMS", cache_key)
    if svis is None:
        st.error("❌ OptionMarket not found in cache. Please calibrate first on the home page.")
        st.stop()
    else:
        available_dates = svis.keys()
        maturity = st.selectbox("Available Dates :", available_dates)

        if st.button("🧮 Calibrate SVI", key="svi_btn"):
            try:
                om = constants.get_from_cache("OptionMarket", cache_key)
                if om is None:
                    om = OptionMarket("Black-Scholes-Merton", constants.FILE_PATH, constants.FILE_UNDERLYING, pricing_date)

                types, strikes, prices, spot, T, options = om.get_values_for_calibration_SVI(pricing_date, maturity)
                st.success(f"Loaded from calibration {len(strikes)} options | Spot: {spot:.2f} | T = {T:.2f}y")
                
                spot = st.session_state.spot
                st.metric(label="📈 Spot on valuation date", value=f"{spot:.2f}")

                # Retrieve SVI parameters from cache
                svi_params = svis.get(maturity) if svis else None

                if svi_params is None:
                    st.warning("⚠️ SVI parameters not found in cache for the selected maturity.")
                else:
                    a, b, p, m, s = svi_params
                    st.write(f"**SVI Parameters for {maturity}:**")
                    st.json({"a": a, "b": b, "p": p, "m": m, "s": s})

                    # Estimate implied vol from SVI for each strike
                    k_vec = np.log(np.array(strikes) / spot)
                    w_vec = a + b * (p * (k_vec - m) + np.sqrt((k_vec - m)**2 + s**2))
                    implied_vols = np.sqrt(w_vec / T)

                    df_result = pd.DataFrame({
                        "Strike": strikes,
                        "Market Price": prices,
                        "Log-Moneyness": k_vec,
                        "Implied Vol (SVI)": implied_vols
                    })
                    st.dataframe(df_result)

                    # Plot vol smile
                    fig, ax = plt.subplots()
                    ax.plot(strikes, implied_vols, label='SVI Implied Vol', color='green')
                    ax.set_title("SVI Implied Volatility by Strike")
                    ax.set_xlabel("Strike")
                    ax.set_ylabel("Implied Volatility")
                    ax.grid(True)
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"❌ Error during SVI calibration: {e}")

with ssvi_tab:
    st.header("SSVI Volatility Surface")
    if st.button("📊 Calibrate SSVI", key="ssvi_btn"):
        try:
            ssvi = SSVICalibration("Black-Scholes-Merton", constants.FILE_PATH, constants.FILE_UNDERLYING, pricing_date)
            params = ssvi.get_ssvi_params()
            st.success("✅ SSVI Calibrated Successfully")
            st.json(params)
        except Exception as e:
            st.error(f"❌ Error during SSVI calibration: {e}")

with dupire_tab:
    st.header("Dupire Local Volatility Surface")
    if st.button("🔍 Calibrate Dupire Local Vol", key="dupire_btn"):
        try:
            dupire = DupireLocalVol("Black-Scholes-Merton", constants.FILE_PATH, constants.FILE_UNDERLYING, pricing_date)
            vol_matrix = dupire.get_implied_vol_matrix()
            st.success("✅ Local Volatility Surface Ready")
            st.dataframe(vol_matrix)
        except Exception as e:
            st.error(f"❌ Error during Dupire calibration: {e}")
