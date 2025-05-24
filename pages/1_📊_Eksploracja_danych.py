import streamlit as st

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import joblib

import functions.boxes as boxes
import functions.plots as plots
import functions.models as models

# --- PARAMETRY ---
path = "datasets/hack/paczkomaty.json"
model_path = "model.joblib"

# ------------------------

st.set_page_config(layout="wide")

# Title
st.title("Eksploracja danych")

# Header
st.header("Wybór lokalizacji")

# Text input
city = st.selectbox(
    "Wybierz miasto do wyświetlenia",
    options=["Białystok", "Warszawa", "Lublin"]
)

df = pd.read_json(path)
df = pl.from_pandas(df)
struct_columns = [col_name for col_name, dtype in df.schema.items() if isinstance(dtype, pl.Struct)]
df = df.unnest(struct_columns)

df_city, grid_gdf, grid_with_counts = boxes.create_city_grid(df, city)
model = joblib.load(model_path)

fig_plotly = plots.plot_city(df_city, grid_gdf, grid_with_counts)

# Wyświetlanie mapy
st.title("Mapa miasta")
st.plotly_chart(fig_plotly, use_container_width=True)

