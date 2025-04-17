#-----------------------------------------------------------------------------------------------------
#------------------------------------SETUP de L'APP---------------------------------------------------
#-----------------------------------------------------------------------------------------------------
import subprocess
import sys
import os

#Récupération du chemin absolu de l'app
APP_DIR = os.path.dirname(os.path.abspath(__file__))
#Chemins pour les assets
ASSETS_DIR = os.path.join(APP_DIR, "assets")
STYLE_PATH = os.path.join(ASSETS_DIR, "style.css")
LOGO_PATH = os.path.join(ASSETS_DIR, "dauphine_logo.png")

import streamlit as st
from streamlit.components.v1 import html
from PIL import Image
import base64
import time
from scripts.products import SSVICalibration, DupireLocalVol, HestonHelper
import constants

#Mise en Page de la page principale.
st.session_state.STYLE_PATH = STYLE_PATH
st.set_page_config(page_title="Pricer produits structurés", layout="wide")  # Active le mode large

def apply_css(STYLE_PATH):
    with open(STYLE_PATH, "r") as f:
        css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
        
apply_css(STYLE_PATH)
logo = Image.open(LOGO_PATH)

#Chargement + affichage logo dauphine.
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
logo_base64 = get_base64_image(LOGO_PATH)
st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{logo_base64}" style="width: 400px;">
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<hr>", unsafe_allow_html=True)

# ---------- STYLE & CENTRAGE ----------
st.markdown("""
    <div style='text-align:center;'>
        <p style='color:#002060; font-size:16px; padding:10px; border:1px solid #002060; border-radius:8px; display:inline-block;'>
            En raison du nombre de données limitées pour calibrer les modèles de volatilité, <br>
            veuillez choisir une des trois dates suivantes, puis cliquez sur <b>calibrer</b>.
        </p>
    </div>
""", unsafe_allow_html=True)

# ---------- FORMULAIRE ----------
with st.container():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
        available_dates = ["13/03/2025", "14/03/2025"] #"12/03/2025"
        selected_date = st.selectbox("Dates disponibles :", available_dates)
        st.session_state["PRICING_DATE"] = selected_date

        # Centrage horizontal du bouton de calibration
        st.markdown("<div style='display:flex; justify-content:center; margin-top:10px;'>", unsafe_allow_html=True)

        if st.button("⚙️ Calibrer les modèles"):
            with st.spinner("⏳ Calibration en cours... Cela peut prendre un peu de temps..."):

                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.markdown("<span style='color:#002060;'>Étape 1/3 : Calibration SSVI en cours...</span>", unsafe_allow_html=True)
                ssvi = SSVICalibration(model="Black-Scholes-Merton",
                                    data_path=constants.FILE_PATH,
                                    file_name_underlying=constants.FILE_UNDERLYING,
                                    pricing_date=selected_date)
                progress_bar.progress(33)

                status_text.text("<span style='color:#002060;'>Étape 2/3 : Calibration volatilité locale en cours...")
                dupire = DupireLocalVol(model="Black-Scholes-Merton",
                                        data_path=constants.FILE_PATH,
                                        file_name_underlying=constants.FILE_UNDERLYING,
                                        pricing_date=selected_date)
                progress_bar.progress(66)

                """
                status_text.text("<span style='color:#002060;'>Étape 3/3 : Calibration Heston en cours...")
                heston = HestonHelper(model="Black-Scholes-Merton",
                                    data_path=constants.FILE_PATH,
                                    file_name_underlying=constants.FILE_UNDERLYING,
                                    pricing_date=selected_date)
                """
                progress_bar.progress(100)
                st.markdown(
                    '<div class="success-box">✅ Calibration terminée avec succès.</div>',
                    unsafe_allow_html=True
                )

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        if st.button("🧹 Réinitialiser les paramètres en cache. Attention à recalibrer les modèles !"):
            from constants import clear_cache
            clear_cache()
            st.success("Cache vidé.")