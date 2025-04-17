import streamlit as st
from constants import DICT_PRODUCT, FILE_PATH, FILE_UNDERLYING, FORMAT_DATE
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


st.title("📦 Construction du portefeuille d’options")
st.metric(label="📈 Spot à la date de valorisation", value=f"{spot:.2f}")

col1, col2 = st.columns(2)
with col1:
    selected_type = st.selectbox("Type d’option", DICT_PRODUCT.keys())
with col2:
    quantity = st.number_input("Quantité", min_value=1, value=1)

col1, col2 = st.columns(2)
with col1:
    st.text_input("Date de début", value=start_date, disabled=True)
with col2:
    end_date = st.date_input("Date de fin")

col1, col2 = st.columns(2)
with col1:
    strike = st.number_input("Strike", value=210.0)
with col2:
    barrier = None
    if "Barrier" in selected_type:
        barrier = st.number_input("Strike barrière", value=180.0 if "Down" in selected_type else 240.0)

model = st.selectbox("Modèle", ["Black-Scholes-Merton", "Heston", "Dupire"])

# Ajouter au portefeuille
if st.button("➕ Ajouter cette option au portefeuille"):
    st.session_state.portfolio._add_product(
        type_product=selected_type,
        start_date=start_date,
        end_date=end_date.strftime(FORMAT_DATE),
        quantity=quantity,
        strike=strike,
        barrier_strike=barrier,
        model=model,
        notional=1
    )

    # Ajout dans le tableau affiché
    st.session_state.portfolio_data.append({
        "Type": selected_type,
        "Modèle": model,
        "Début": start_date,
        "Fin": end_date.strftime(FORMAT_DATE),
        "Strike": strike,
        "Barrière": barrier,
        "Quantité": quantity
    })

    st.success("✅ Option ajoutée au portefeuille")

# Affichage du portefeuille
if len(st.session_state.portfolio_data) > 0:
    st.markdown("### 🧾 Portefeuille actuel")
    st.dataframe(st.session_state.portfolio_data, use_container_width=True)
else:
    st.info("Aucune option dans le portefeuille pour l’instant.")