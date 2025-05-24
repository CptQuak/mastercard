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
X_train_path = "X_train.joblib"
y_train_path = "y_train.joblib"

# ------------------------

st.set_page_config(layout="wide")

st.title("Predykcje")

# Header
st.header("Wybór lokalizacji")

# City selectbox
city = st.selectbox(
    "Wybierz miasto do wyświetlenia",
    options=["Białystok", "Warszawa", "Lublin"]
)

# Header
st.header("Wybór budżetu")

# Budget input
budget = st.number_input("Wybierz budżet", min_value=1, max_value=100000, step=1)

model = joblib.load(model_path)
X_train = joblib.load(X_train_path)
y_train = joblib.load(y_train_path)

df = pd.read_json(path)
df = pl.from_pandas(df)
struct_columns = [col_name for col_name, dtype in df.schema.items() if isinstance(dtype, pl.Struct)]
df = df.unnest(struct_columns)

df_city, grid_gdf, grid_with_counts = boxes.create_city_grid(df, 'Lublin')
predicts = models.model_predict(model, df_city, grid_gdf, grid_with_counts)

fig_plotly = plots.plot_city(df_city, grid_gdf, predicts, 'target')


# Create two columns (panels)
left_col, right_col = st.columns([0.7, 1.3])

# Left Panel Content
with left_col:
    st.header("Informacje o predykcji")
    preds = models.explain_prediction(grid_gdf, 1, model, X_train, y_train)
    
    # Create the figure
    fig, ax = plt.subplots()

    # Temporary margin on top
    st.markdown("<br>", unsafe_allow_html=True)

    # Plot your data
    ax.plot(preds.result['cumulative'], preds.result['variable_name'], '-')

    # Show in Streamlit
    st.pyplot(fig)

# Right Panel Content
with right_col:
    st.header("Mapa miasta")
    st.plotly_chart(fig_plotly, use_container_width=True)