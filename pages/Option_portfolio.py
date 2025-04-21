import streamlit as st
from constants import DICT_PRODUCT_S,FORMAT_DATE, BASE_DIV_RATE, BASE_SPOT_RANGE, BASE_STEPS_GREEKS,BASE_VOL_RANGE
from scripts.products import Portfolio
from scipy.signal import savgol_filter
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
#-----------------------------------------------------------------------------------------------------
#------------------------------------Page construction du portefeuille---------------------------------
#-----------------------------------------------------------------------------------------------------
#______________________________________INITIALISATION PAGE____________________________________________
STYLE_PATH=st.session_state.STYLE_PATH
def apply_css(STYLE_PATH):
    with open(STYLE_PATH, "r") as f:
        css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
apply_css(STYLE_PATH)

tab1, tab2 = st.tabs(["📋 Build & Manage Portfolio", "💰 Pricing Results"])
# Shared elements
start_date = st.session_state.pricing_date
spot = st.session_state.spot

# -- INIT portfolio in session if not present --
if "portfolio" not in st.session_state:
    st.session_state.portfolio = Portfolio()
    st.session_state.portfolio_data = []
    st.session_state.portfolio_priced = False
#______________________________________TAB 1: Construction ptf____________________________________________
with tab1:
    st.title("📦 Options Portfolio Builder")
    st.metric(label="📈 Spot on valuation date", value=f"{spot:.2f}")

    col1, col2 = st.columns(2)
    with col1:
        selected_type = st.selectbox("Option Type", DICT_PRODUCT_S.keys())
        quantity = st.number_input("Quantity", value=1)
        strike = st.number_input("Strike", value=round(spot*1.2, 0))
        div_rate = st.number_input("Dividend Yield (%)", value=BASE_DIV_RATE)
    with col2:
        model = st.selectbox("Pricing Model", ["Heston", "Local Volatility (Dupire)"])
        if model == "Local Volatility (Dupire)":
            model = "Dupire"
        end_date = st.date_input("Maturity Date")
        notional = st.number_input("Notional", value=1)
        barrier = None
        barrier_keywords = ["UP", "DOWN", "IN", "OUT"]
        if any(k in selected_type.upper() for k in barrier_keywords):
            barrier = st.number_input("Barrier Strike", value=round(spot*0.8, 0) if "Down" in selected_type else round(spot*1.2, 0))

    # Ajouter au portefeuille
    if st.button("➕ Add this option to portfolio"):
        st.session_state.portfolio._add_product(
            type_product=selected_type,
            start_date=start_date,
            end_date=end_date.strftime(FORMAT_DATE),
            quantity=quantity,
            strike=strike,
            barrier_strike=barrier,
            model=model,
            div_rate=div_rate,
            notional=notional,
        )

        # Ajout dans le tableau affiché
        st.session_state.portfolio_data.append({
            "Type": selected_type,
            "Pricing Model": model,
            "Start": start_date,
            "Maturity": end_date.strftime(FORMAT_DATE),
            "Strike": strike,
            "Barrier": barrier,
            "Quantity": quantity,
            "Dividend Yield": div_rate,
            "Notional":notional,
        })

        st.success("✅ Option added to portfolio !")

    st.markdown("### 🧾 Portfolio Contents")
    # Affichage du portefeuille
    portfolio_placeholder = st.empty()
    if len(st.session_state.portfolio_data) > 0:
        portfolio_placeholder.dataframe(st.session_state.portfolio_data, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧹 Clear Portfolio"):
                st.session_state.portfolio.clear_portfolio()
                st.session_state.portfolio_data = []
                st.session_state.portfolio_priced = False
                portfolio_placeholder.empty()
        with col2:
            if st.button("💸 Price Portfolio"):
                st.session_state.npv, st.session_state.ptf_npvs, st.session_state.payoffs, st.session_state.spots = st.session_state.portfolio.price_portfolio()
                st.session_state.portfolio_priced = True
                st.success("✅ Portfolio priced. See results in the second tab.")
    else:
        st.info("No options added yet.")

#______________________________________TAB 2: Pricing and Greeks__________________________________________
with tab2:
    if st.session_state.get("portfolio_priced", False):
        st.title("💰 Portfolio Pricing Results")
        st.metric("📊 Net Present Value (NPV)", f"{st.session_state.npv:.2f}")

        # Filter table and add NPV row
        cols_to_display = ["Type", "Strike", "Barrier", "Quantity", "Notional"]

        df = pd.DataFrame(st.session_state.portfolio_data)
        df = df[cols_to_display].copy()

        # Add NPV column (per product)
        df["Product NPV"] = st.session_state.ptf_npvs

        # Add a row for total portfolio NPV
        npv_row = pd.Series(["Total NPV", "", "", "", "", st.session_state.npv], index=df.columns)
        df = pd.concat([df, pd.DataFrame([npv_row])], ignore_index=True)

        st.markdown("### 📋 Priced Portfolio Breakdown")
        st.dataframe(df, use_container_width=True)

        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            # Plot payoff vs. spot
            st.markdown("### 📈 Payoff vs. Spot")

            fig = px.scatter(
                x=st.session_state.spots,
                y=st.session_state.payoffs,
                labels={"x": "Spot", "y": "Payoff"},
                title="Portfolio Payoff at Maturity",
            )

            fig.update_traces(marker=dict(size=6, color='#002060'))  # Dauphine blue
            fig.update_layout(height=500)

            st.plotly_chart(fig, use_container_width=True)

        # --- Greeks Table and Portfolio Interpretation ---
        st.subheader("📉 Option Greeks Breakdown")

        portfolio_data = st.session_state.portfolio_data
        ptf = st.session_state.portfolio
        pricers = ptf._portfolio

        greek_table = []
        ptf_greeks = {"Delta": 0, "Gamma": 0, "Vega": 0, "Theta": 0, "Rho": 0}

        for i, ((prod_type, model, strike, start, end, barrier, div, rate, dc, rc, notional, fmt, cur, sigma), entry) in enumerate(pricers.items()):
            q = entry["quantity"]
            pricer = entry["pricer"]
            payoff_matrix = np.array(pricer._payoff)
            exec_prob = (payoff_matrix > 0).sum() / payoff_matrix.size

            greek_row = {
                "Product": f"{prod_type} ({model})",
                "Strike": round(float(strike), 2),
                "Delta": round(float(pricer.delta * q), 4),
                "Gamma": round(float(pricer.gamma * q), 4),
                "Vega": round(float(pricer.vega * q), 4),
                "Theta": round(float(pricer.theta * q), 4),
                "Rho": round(float(pricer.rho * q), 4),
                "Execution Probability %": round(float(exec_prob)*100, 4)
            }
            for g in ptf_greeks:
                ptf_greeks[g] += greek_row[g]
            greek_table.append(greek_row)

        # Append portfolio total
        greek_table.append({
            "Product": "📦 Portfolio Total",
            "Strike": None,
            "Delta": ptf_greeks["Delta"],
            "Gamma": ptf_greeks["Gamma"],
            "Vega": ptf_greeks["Vega"],
            "Theta": ptf_greeks["Theta"],
            "Rho": ptf_greeks["Rho"],
            "Execution Probability %": None
        })

        greek_df = pd.DataFrame(greek_table)

        # Layout: two columns
        left_col, right_col = st.columns([2, 2])

        with left_col:
            st.markdown("### Portfolio Greeks Summary")
            st.dataframe(greek_df, use_container_width=True)

        # Extract total row for explanations
        total = greek_df[greek_df["Product"] == "📦 Portfolio Total"].iloc[0]

        delta = total["Delta"]
        gamma = total["Gamma"]
        vega = total["Vega"]
        theta = total["Theta"]
        rho = total["Rho"]
        theta_daily = round(theta / 252, 4)

        with right_col:
            st.markdown(f"""
            <div style='
                background-color: #ffffff;
                border: 2px solid #002060;
                padding: 1rem 1.2rem;
                border-radius: 0.5rem;
                font-size: 0.95rem;
                line-height: 1.6;
                margin-top: 0.5rem;
            '>
            <h4 style='margin-top: 0; color: #002060;'>📘 Portfolio Greeks Interpretation</h4>

            - 🔺 <strong>Delta:</strong> <code>{delta:.4f}</code> → Portfolio value increases by €{delta:.2f} per €1 move in the underlying.<br>
            - 🔼 <strong>Gamma:</strong> <code>{gamma:.6f}</code> → Delta changes with spot.<br>
            - 📈 <strong>Vega:</strong> <code>{vega:.4f}</code> → Portfolio gains €{vega:.2f} per +1% change in implied volatility.<br>
            - ⏳ <strong>Theta:</strong> <code>{theta:.2f}</code> → Annual time decay. Approx. <strong>€{theta_daily:.4f}</strong> loss <i>per day</i> if nothing changes.<br>
            - 💸 <strong>Rho:</strong> <code>{rho:.2f}</code> → Value moves €{rho / 100:.2f} per +1% change in interest rates.

            </div>
            """, unsafe_allow_html=True)

                # --- Greeks vs Spot Graphs (Portfolio) ---
        st.subheader("📈 Portfolio Greeks Sensitivity vs Spot")
        
        # Button to trigger computation
        if st.button("📊 Generate Greeks Charts / PnL Matrix"):
            progress_text = "Computing portfolio Greeks vs Spot..."
            progress_bar = st.progress(0, text=progress_text)

            greek_charts = []
            total_greeks = ["delta", "gamma", "vega", "theta", "rho"]

            def compute_portfolio_greek_curve(ptf, greek_name="delta", spot_range=BASE_SPOT_RANGE, steps=BASE_STEPS_GREEKS):
                #base_spot = np.array(st.session_state.spot * np.ones(len(ptf._portfolio.values())))
                base_spot = float(st.session_state.spot)
                lower = base_spot * (1+spot_range)
                upper = base_spot * (1-spot_range)
                spot_points = np.linspace(lower, upper, steps)

                greek_vals = []
                for spot in spot_points:
                    greek_total = 0
                    for _, entry in ptf._portfolio.items():
                        opt = entry["pricer"].__deep_copy__()
                        opt._spot = float(spot)
                        greek_total += getattr(opt, greek_name) * entry["quantity"]
                    greek_vals.append(greek_total)
                values_greek = [float(x) for x in greek_vals]
                spot_points = [float(x) for x in spot_points]
                values_greek = savgol_filter(values_greek, BASE_STEPS_GREEKS, 3)
                return pd.DataFrame(list(zip(spot_points, values_greek)), columns=["Spot", greek_name.capitalize()])

            # Compute all charts with a progress bar
            for i, greek in enumerate(total_greeks):
                df_curve = compute_portfolio_greek_curve(ptf, greek)
                fig = px.line(df_curve, x="Spot", y=greek.capitalize(), title=f"{greek.capitalize()} vs Spot Price (Portfolio)")
                current_spot = st.session_state.spot
                current_val = ptf_greeks[greek.capitalize()]
                fig.add_scatter(x=[current_spot], y=[current_val], mode="markers", name="Current", marker=dict(color="black", size=8))
                greek_charts.append(fig)

                progress_bar.progress((i + 1) / len(total_greeks), text=f"Completed {greek.capitalize()}")

            progress_bar.empty()
            st.success("✅ All Greeks Sensitivity charts generated!")

            # First two rows with 2 charts each
            for row_start in range(0, 4, 2):
                cols = st.columns(2)
                for i in range(2):
                    if row_start + i < len(greek_charts):
                        with cols[i]:
                            st.plotly_chart(greek_charts[row_start + i], use_container_width=True)

            # Last row with 1 chart, centered
            if len(greek_charts) == 5:
                cols = st.columns([1, 6, 1])  # Center the last chart
                with cols[1]:
                    st.plotly_chart(greek_charts[4], use_container_width=True)

            # --- PnL Matrix (Spot x Volatility) ---
            st.subheader("💹 Portfolio P&L Matrix (vs Spot & Volatility)")
            spot_center = st.session_state.spot
            spot_steps = np.linspace(spot_center * 0.8, spot_center * 1.2, 10)
            @st.cache_data
            def compute_pnl_matrix(_ptf, spots, vol_shocks):
                base_price = st.session_state.npv
                pnl_matrix = []

                for vol in vol_shocks:
                    row = []
                    for spot in spots:
                        price = 0
                        for _, entry in _ptf._portfolio.items():
                            pricer = entry["pricer"].__deep_copy__()
                            quantity = entry["quantity"]

                            # --- Apply volatility shocks based on model type ---
                            if pricer._model_name == "Heston":
                                if vol < 0:
                                    pricer._model._v0 = pricer._model._v0-(vol**2)
                                    #pricer._model._theta = pricer._model._theta-(vol**2)
                                else:
                                    pricer._model._v0 = pricer._model._v0+vol**2
                                    #pricer._model._theta = pricer._model._theta+vol**2
                                pricer._spot = spot
                                price += pricer.price * quantity
                            elif pricer._model_name == "Dupire":
                                pricer._spot = spot
                                pricer._model._spread_vol=vol
                                price += pricer.price * quantity
                            else:
                                pass
                        row.append(round(price - base_price, 2))
                    pnl_matrix.append(row)

                return np.array(pnl_matrix), base_price

            pnl_matrix, base_val = compute_pnl_matrix(ptf, spot_steps, BASE_VOL_RANGE)

            z_rounded = np.round(pnl_matrix, 2)
            text_labels = [[f"{val:.2f}" for val in row] for row in z_rounded]

            fig = go.Figure(data=go.Heatmap(
                z=z_rounded,
                text=text_labels,
                texttemplate="%{text}",
                textfont={"size":12},
                x=[f"{s:.0f}" for s in spot_steps],
                y=[f"{v*100:.0f}%" for v in BASE_VOL_RANGE],
                colorscale="RdYlGn",
                colorbar=dict(title="PnL (€)", ticksuffix=" €"),
                zmid=0,
                hovertemplate="Spot: %{x}<br>Vol: %{y}<br>PnL: %{z} €<extra></extra>"
            ))

            fig.update_layout(
                title="PnL Matrix (Δ NPV from current)",
                xaxis_title="Spot Price",
                yaxis_title="Implied Volatility",
                height=500,
                margin=dict(t=40, l=60, r=60, b=60)
            )

            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("⚠️ Price the portfolio first in the previous tab.")