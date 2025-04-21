import streamlit as st
from constants import FORMAT_DATE, BASE_DIV_RATE, BASE_NOTIONAL, BASE_CURRENCY, BASE_RATE, AutocallsType, Types
from scripts.products import AutocallPricer
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np

STYLE_PATH = st.session_state.STYLE_PATH

def apply_css(STYLE_PATH):
    with open(STYLE_PATH, "r") as f:
        css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
apply_css(STYLE_PATH)

start_date = st.session_state.pricing_date
spot = st.session_state.spot

st.title("🔁 Autocalls Pricing Engine")
tab1, tab2 = st.tabs(["🏗️ Build & Price", "📊 Results"])

# ---- SESSION INIT ----
if "autocall_pricer" not in st.session_state:
    st.session_state.autocall_pricer = None
    st.session_state.autocall_results = {}

with tab1:
    st.header("🏗️ Define Autocall Structure")
    st.metric("📈 Spot on valuation date", f"{spot:.2f}")

    st.subheader("Main Parameters")
    col1 = st.columns(2)
    with col1[0]:
        end_date = st.date_input("Maturity Date")
        autocall_type = st.selectbox("Autocall Type", ["Vanilla", "Phoenix"])
        if autocall_type == "Vanilla":
            autocall_type = AutocallsType.AUTOCALL
        else:
            autocall_type = AutocallsType.PHOENIX


    col_main1, col_main2 = st.columns(2)
    with col_main1:
        strike = st.number_input("Autocall Strike (%)", value=120.0)
    with col_main2:
        final_strike = st.number_input("Final Autocall Strike (%)", value=120.0)

    col1, col2 = st.columns(2)
    with col1:
        coupon = st.number_input("Coupon Rate (%)", value=8.0)
        coupon_strike = st.number_input("Coupon Barrier (%)", value=70.0)
        protection = st.number_input("Capital Protection Barrier (%)", value=60.0)
    with col2:
        exercise_type = st.selectbox("Exercise Type", ["American", "European"])
        if exercise_type != "American":
            frequency = st.selectbox("Coupon Frequency", ["monthly", "quarterly", "semi-annually", "annually"])
        else:
            frequency = None

        model = st.selectbox("Pricing Model", ["Heston", "Local Volatility (Dupire)"])
        if model == "Local Volatility (Dupire)":
            model = "Dupire"

        memory = False
        if autocall_type == AutocallsType.PHOENIX:
            memory = st.checkbox("Coupon Memory", value=True)

    if st.button("💸 Price Autocall"):
        pricer = AutocallPricer(
            start_date=start_date,
            end_date=end_date.strftime(FORMAT_DATE),
            strike=strike / 100,
            final_strike=final_strike / 100,
            coupon=coupon / 100,
            coupon_strike=coupon_strike / 100,
            protection=protection / 100,
            frequency=frequency,
            memory=memory,
            exercise_type=Types.AMERICAN if exercise_type == "American" else Types.EUROPEAN,
            model=model,
            rate=BASE_RATE,
            div_rate=BASE_DIV_RATE,
            notional=BASE_NOTIONAL,
            currency=BASE_CURRENCY,
            type=AutocallsType.PHOENIX if autocall_type == "Phoenix" else AutocallsType.AUTOCALL,
            spot=spot,
        )
        npv, payoff, par_coupon, call_curve = pricer.price
        st.session_state.autocall_pricer = pricer
        st.session_state.autocall_results = {
            "npv": npv,
            "payoff": payoff,
            "par_coupon": par_coupon,
            "call_curve": call_curve,
        }
        st.session_state.autocall_paths = pricer._option._paths
        st.success("✅ Autocall priced successfully!")

with tab2:
    st.header("📊 Pricing Results")
    if st.session_state.autocall_results:
        results = st.session_state.autocall_results
        st.metric("💰 Net Present Value (NPV)", f"{results['npv']:.2f}")
        st.metric("🎯 Par Coupon (NPV = 0)", f"{results['par_coupon']*100:.2f}%")

        #Paths setup:
        paths = st.session_state.autocall_paths
        time = paths['time']
        paths_spots = paths['Spots'][:5]
        
        # Plot Paths
        fig, ax = plt.subplots(figsize=(12, 6))
        for i in range(len(paths_spots)):
            ax.plot(time, paths_spots[i], label=f"Path {i+1}")

        #Plot barriers:
        barriers = strike/100*spot*np.ones(len(time))
        ax.plot(time, barriers, 'k--', label="Autocall Barrier")

        coupon_barriers = coupon_strike/100*spot*np.ones(len(time))
        if not np.array_equal(barriers[:len(time)], coupon_barriers[:len(time)]):
            ax.plot(time, coupon_barriers, 'g--', label="Coupon Barrier")

        protec = protection/100*spot*np.ones(len(time))
        if not np.array_equal(barriers[:len(time)], protec[:len(time)]):
            ax.plot(time, protec, color='red', linestyle='--', label="Protection Barrier")
        ax.plot(time, final_strike/100*spot*np.ones(len(time)), label="Final Strike", linestyle='--', color='purple')
        
        all_y = np.array(paths_spots).flatten()
        y_min, y_max = all_y.min(), all_y.max()
        ax.set_ylim([y_min * 0.5, y_max * 1.2])

        ax.set_title("Simulated Asset Price Paths with Autocall Features")
        ax.set_xlabel("Time (Years)")
        ax.set_ylabel("Asset Price")
        ax.grid(True)
        ax.legend()
        st.plotly_chart(fig)

        # Call Probability Curve
        st.subheader("📉 Call Probability Curve")
        curve_df = pd.DataFrame(results["call_curve"], columns=["Time (Years)", "Cumulative Call Probability"])
        fig_curve = px.line(curve_df, x="Time (Years)", y="Cumulative Call Probability",
                            markers=True, title="Cumulative Call Probability Over Time")
        fig_curve.update_layout(yaxis=dict(range=[0, curve_df["Cumulative Call Probability"].max() * 1.3]))
        st.plotly_chart(fig_curve, use_container_width=True)
    else:
        st.info("⚠️ Please price an autocall first in the previous tab.")
