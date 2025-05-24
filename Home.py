import streamlit as st

st.set_page_config(layout="wide")

# Centered layout using HTML
st.markdown(
    """
    <div style='text-align: center; padding-top: 100px;'>
        <h1 style='font-size: 3.5em;'>📦 Paczkomat Wizard 🧙‍♂️</h1>
        <p style='font-size: 1.5em;'> ▶ Twoje narzędzie do strategicznego rozmieszczania paczkomatów ◀ </p>
        <p style='font-size: 1.5em; color: gray;'>Version 1.0.0</p>
    </div>
    """,
    unsafe_allow_html=True
)
