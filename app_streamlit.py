"""Interfață modernă în Streamlit pentru AgriVision AI Classifier."""

import requests
import streamlit as st

# Configurare pagină
st.set_page_config(
    page_title="AgriVision AI Dashboard", page_icon="🌾", layout="centered"
)

# Configurare pagină
st.set_page_config(
    page_title="AgriVision AI Dashboard", page_icon="🌾", layout="centered"
)

st.title("🌾 AgriVision AI Classifier")
st.subheader("Modern Data Science Dashboard")
st.write(
    "Selectează o imagine din exploratorul de fișiere pentru analiză în timp real."
)

# Cutie nativă de încărcare fișiere
uploaded_file = st.file_uploader(
    "Alege o imagine cu un fruct...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Afișarea automată a previzualizării imaginii
    st.image(uploaded_file, caption="Imagine încărcată pentru analiză", width=250)

    if st.button("Rulează Predicția Keras", type="primary"):
        with st.spinner("Se procesează imaginea prin rețeaua MobileNetV2..."):
            try:
                # Pregătim fișierul pentru trimitere către serverul Flask existent
                files = {
                    "image": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type,
                    )
                }
                response = requests.post(
                    "http://localhost:5000/predict", files=files, timeout=10
                )
                data = response.json()

                if data["status"] == "success":
                    st.success(f"📊 Clasa Detectată: {data['predicted_class'].upper()}")
                    st.metric(
                        label="Scor Confidență",
                        value=f"{data['confidence'] * 100:.2f}%",
                    )

                    # Generarea automată a graficului cu procente (O singură linie de cod!)
                    st.write("**Distribuția Probabilităților:**")
                    st.bar_chart(data["distribution"])
                else:
                    st.error(f"Eroare API: {data['error_message']}")
            except requests.exceptions.ConnectionError:
                st.error(
                    "Nu s-a putut conecta la serverul Flask. Asigură-te că app.py rulează activ pe portul 5000!"
                )
