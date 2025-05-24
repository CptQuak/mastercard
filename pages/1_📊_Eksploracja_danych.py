import streamlit as st

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import joblib

from functions import boxes
from functions import plots
from functions import models, features

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
    options=["Białystok", "Lublin"]
)

# Reading dataset
df = pd.read_json(path)
df = pl.from_pandas(df)
struct_columns = [col_name for col_name, dtype in df.schema.items() if isinstance(dtype, pl.Struct)]
df = df.unnest(struct_columns)

# Creating city grid
df_city, grid_gdf, grid_with_counts = boxes.create_city_grid(df, city)

# Loading model
model = joblib.load(model_path)

# Plotting cities
fig_plotly = plots.plot_city(df_city, grid_gdf, grid_with_counts)

# Map display
st.title("Mapa miasta")
st.plotly_chart(fig_plotly, use_container_width=True)

