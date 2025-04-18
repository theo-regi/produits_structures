import streamlit as st
from constants import DICT_PRODUCT_S, FILE_PATH, FILE_UNDERLYING, FORMAT_DATE, BASE_DIV_RATE, BASE_NOTIONAL
from scripts.products import Portfolio, OptionMarket
import pandas as pd
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------------------------------
#------------------------------------Page construction du portefeuill---------------------------------
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

        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            # Plot payoff vs. spot
            st.markdown("### 📈 Payoff vs. Spot")

            fig, ax = plt.subplots()
            ax.scatter(st.session_state.spots, st.session_state.payoffs, label='Payoff', s=10)
            ax.set_xlabel('Spot')
            ax.set_ylabel('Payoff')
            ax.set_title('Portfolio Payoff at Maturity')
            ax.grid(True)
            st.pyplot(fig)

    else:
        st.info("⚠️ Price the portfolio first in the previous tab.")