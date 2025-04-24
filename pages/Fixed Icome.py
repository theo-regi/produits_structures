import streamlit as st
from scripts.products import ZCBond, FixedLeg, FloatLeg, Swap
from scripts.utils import Rates_curve, get_market
from constants import FORMAT_DATE, CURVE_PATH, BASE_NOTIONAL, CONVENTION_DAY_COUNT, ROLLING_CONVENTION, TYPE_INTERPOL, EXCHANGE_NOTIONAL

import pandas as pd
from datetime import date
import plotly.express as px
import os

STYLE_PATH = st.session_state.STYLE_PATH
# ---------------- INIT -----------------
st.set_page_config(layout="wide")
st.title("🏦 Fixed Income Product Builder")

def apply_css(STYLE_PATH):
    with open(STYLE_PATH, "r") as f:
        css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
apply_css(STYLE_PATH)


curve = Rates_curve(path_rate=CURVE_PATH)
st.subheader("🎯 Build a Fixed Income Product")

# Step 1: Product selector
product_type = st.selectbox("Choose Product Type", ["Zero Coupon Bond", "Fixed Leg", "Floating Leg", "Swap"])

# ZC Bond Block
if product_type == "Zero Coupon Bond":
    with st.form("zc_bond_form"):
        st.markdown("#### 🔹 Zero Coupon Bond Inputs")
        notional = st.number_input("Notional", value=BASE_NOTIONAL, key="zc_notional")
        maturity = st.number_input("Maturity (Years)", value=2.0, step=0.25)
        rate = st.number_input("Zero Rate (%)", value=3.0, step=0.1)

        submitted = st.form_submit_button("📥 Price ZC Bond")
        if submitted:
            zc = ZCBond(notional)
            df = zc.get_discount_factor_from_zcrate(rate / 100, maturity)
            price = zc.get_npv_zc_from_df(df)

            st.metric("NPV", f"{price:,.2f}")
            st.metric("YTM", f"{zc.get_ytm(price, maturity):.4%}")
            st.metric("Duration McAuley", f"{zc.get_duration_macaulay(maturity):.2f}")
            st.metric("Modified Duration", f"{zc.get_modified_duration(price, maturity):.2f}")
            st.metric("Convexity", f"{zc.get_convexity(maturity, market_price=price):.2f}")

# Fixed Leg Block
elif product_type == "Fixed Leg":
    with st.form("fixed_leg_form"):
        st.markdown("#### 🔹 Fixed Leg Inputs")
        currency = st.selectbox("Currency", ["EUR", "USD", "GBP", "BRL"], index=0, key="fixed_currency")
        notional = st.number_input("Notional", value=BASE_NOTIONAL, key="fixed_notional")
        start_date = st.date_input("Start Date", date.today(), key="fixed_start")
        end_date = st.date_input("End Date", date.today().replace(year=date.today().year + 2), key="fixed_end")
        freq = st.selectbox("Payment Frequency", ["monthly", "quarterly", "semi-annually", "annually"], index=2)
        rate = st.number_input("Coupon Rate (%)", value=3.5, step=0.1)

        submitted = st.form_submit_button("📥 Price Fixed Leg")
        if submitted:
            curve_pricing = Rates_curve(CURVE_PATH, rate)
            leg = FixedLeg(
                rate_curve=curve_pricing,
                start_date=start_date.strftime(FORMAT_DATE),
                end_date=end_date.strftime(FORMAT_DATE),
                discounting_curve=curve,
                paiement_freq=freq,
                currency=currency,
                notional=notional,
                exchange_notional=True
            )
            npv = leg.calculate_npv(leg._cashflows)

            st.metric("NPV", f"{npv:,.2f}")
            st.metric("Duration", f"{leg.calculate_duration():.2f}")
            st.metric("Sensitivity", f"{leg.calculate_sensitivity():.2f}")
            st.metric("Convexity", f"{leg.calculate_convexity():.2f}")

# Floating Leg Block
elif product_type == "Floating Leg":
    with st.form("floating_leg_form"):
        st.markdown("#### 🔹 Floating Leg Inputs")
        currency = st.selectbox("Currency", ["EUR", "USD", "GBP", "BRL"], index=0, key="float_currency")
        notional = st.number_input("Notional", value=BASE_NOTIONAL, key="float_notional")
        start_date = st.date_input("Start Date", date.today(), key="float_start")
        end_date = st.date_input("End Date", date.today().replace(year=date.today().year + 2), key="float_end")
        freq = st.selectbox("Payment Frequency", ["annually"], index=2)
        spread = st.number_input("Spread (bps)", value=25, step=5)

        submitted = st.form_submit_button("📥 Price Floating Leg")
        if submitted:
            floater = FloatLeg(
                rate_curve=curve,
                start_date=start_date.strftime(FORMAT_DATE),
                end_date=end_date.strftime(FORMAT_DATE),
                discounting_curve=curve,
                paiement_freq=freq,
                currency=currency,
                notional=notional,
                spread=spread / 100
            )
            npv = floater.calculate_npv(floater._cashflows)

            st.metric("NPV", f"{npv:,.2f}")
            st.metric("Duration", f"{floater.calculate_duration():.2f}")
            st.metric("Sensitivity", f"{floater.calculate_sensitivity():.2f}")
            st.metric("Convexity", f"{floater.calculate_convexity():.2f}")

elif product_type == "Swap":
    with st.form("swap_form"):
        st.markdown("### 🔹 Swap Product Builder")

        # BASIC LEG PARAMETERS
        st.markdown("#### 🧱 Leg Setup")
        currency = st.selectbox("Currency", ["EUR", "USD", "GBP", "BRL"], key="swap_currency")
        notional = st.number_input("Notional", value=BASE_NOTIONAL, key="swap_notional")
        start_date = st.date_input("Start Date", date.today(), key="swap_start")
        end_date = st.date_input("End Date", date.today().replace(year=date.today().year + 5), key="swap_end")
        freq = st.selectbox("Payment Frequency", ["monthly", "quarterly", "semi-annually", "annually"], index=3)

        # RATE CONVENTIONS & CURVES
        st.markdown("#### ⚙️ Curve & Convention Setup")
        exchange_notional = st.checkbox("Exchange Notional", value=False)

        # COLLAR INPUTS
        with st.expander("💡 Add Optional Collar"):
            use_collar = st.checkbox("Enable Collar Protection")
            cap_strike = st.number_input("Cap Strike", value=0.035, step=0.001, format="%.5f")
            floor_strike = st.number_input("Floor Strike", value=0.02, step=0.001, format="%.5f")
            vol = st.number_input("Implied Volatility (σ)", value=0.06, step=0.01)

        # SUBMIT
        submitted = st.form_submit_button("📥 Price Swap")
        if submitted:
            discount_curve = curve.deep_copy()

            # Construct Swap
            swap = Swap(
                rate_curve=curve,
                start_date=start_date.strftime(FORMAT_DATE),
                end_date=end_date.strftime(FORMAT_DATE),
                paiement_freq=freq,
                currency=currency,
                discounting_curve=discount_curve,
                notional=notional,
                exchange_notional=exchange_notional
            )

            # Price Swap
            fixed_rate = swap.calculate_fixed_rate()

            # Display
            st.metric("Fair Fixed Rate", f"{fixed_rate * 100:.4f} %")
            st.metric("PV01", f"{swap.calculate_pv01():.2f}")
            st.metric("Duration", f"{swap.calculate_duration():.2f}")

            # Collar computation if selected
            if use_collar:
                collar_value = swap.calculate_collar(cap_strike, floor_strike, vol)
                st.metric("Collar Value", f"{collar_value:,.6f}")
            else:
                collar_value = None

