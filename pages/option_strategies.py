import streamlit as st
from constants import DICT_PRODUCT, FILE_PATH, FILE_UNDERLYING, FORMAT_DATE, BASE_DIV_RATE, BASE_NOTIONAL
from scripts.products import Portfolio, OptionMarket

STYLE_PATH=st.session_state.STYLE_PATH
def apply_css(STYLE_PATH):
    with open(STYLE_PATH, "r") as f:
        css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
apply_css(STYLE_PATH)

if "portfolio" not in st.session_state:
    st.session_state.portfolio = Portfolio()
    st.session_state.portfolio_data = []

start_date = st.session_state.pricing_date
omarket = OptionMarket(FILE_PATH, FILE_UNDERLYING)
spot = omarket.get_spot(start_date)


st.title("📦 Options portfolio construction.")
st.metric(label="📈 Actual spot :", value=f"{spot:.2f}")

col1, col2 = st.columns(2)
with col1:
    selected_type = st.selectbox("Option Type", DICT_PRODUCT.keys())
with col2:
    quantity = st.number_input("Quantity", min_value=1, value=1)

col1, col2 = st.columns(2)
with col1:
    st.text_input("Start Date", value=start_date, disabled=True)
with col2:
    end_date = st.date_input("Maturity Date")

col1, col2 = st.columns(2)
with col1:
    strike = st.number_input("Strike", value=210.0)
with col2:
    barrier = None
    if "Barrier" in selected_type:
        barrier = st.number_input("Barrier Strike", value=180.0 if "Down" in selected_type else 240.0)

col1, col2 = st.columns(2)
with col1:
    div_rate = st.number_input("Dividend Yield (%)", value=BASE_DIV_RATE)
with col2:
    notional = st.number_input("Notional", value=BASE_NOTIONAL)

model = st.selectbox("Pricing Models", ["Black-Scholes-Merton", "Heston", "Local Volatility (Dupire)"])

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

# Affichage du portefeuille
if len(st.session_state.portfolio_data) > 0:
    st.markdown("### 🧾 Actual Portfolio:")
    st.dataframe(st.session_state.portfolio_data, use_container_width=True)
else:
    st.info("No option added to portfolio.")