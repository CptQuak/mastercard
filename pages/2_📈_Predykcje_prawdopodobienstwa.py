import streamlit as st

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import joblib
import streamlit.components.v1 as components

from streamlit_plotly_events import plotly_events

from functions import boxes
from functions import plots
from functions import models, features

import warnings
warnings.filterwarnings("ignore")

# --- PARAMETRY ---
path = "datasets/hack/paczkomaty.json"
model_path = "model.joblib"
X_train_path = "X_train.joblib"
y_train_path = "y_train.joblib"
clicked_id = 1

city_dict = {
    "Białystok": "bialystok",
    "Lublin": "lublin"
}
# ------------------------

st.set_page_config(layout="wide")

# Title
st.title("Predykcje")

# Header
st.header("Wybór lokalizacji")

# City selectbox
city = st.selectbox(
    "Wybierz miasto do wyświetlenia",
    options=["Białystok", "Lublin"]
)

# Header
st.header("Wybór ilość paczkomatów")

# Budget input
budget = st.number_input("Wybierz ilość paczkomatów do umieszczenia", min_value=1, max_value=100000, step=1)

# Loading model and train sets
model = joblib.load(model_path)
X_train = joblib.load(X_train_path)
y_train = joblib.load(y_train_path)

# Loading dataset
df = pd.read_json(path)
df = pl.from_pandas(df)
struct_columns = [col_name for col_name, dtype in df.schema.items() if isinstance(dtype, pl.Struct)]
df = df.unnest(struct_columns)

# Creating city grid
city_data = pl.read_parquet(f'{city_dict[city]}.parquet')
df_city, grid_gdf, grid_with_counts = boxes.create_city_grid(df, city)
grid_gdf, grid_with_counts = features.add_hex_features(df_city, grid_gdf, grid_with_counts, city_data)

# Getting predictions
predicts = models.model_predict(model, df_city, grid_gdf, grid_with_counts)

# --- Budget ---
x = grid_with_counts[['id', 'observation_count']].copy()
x['observation_count'] = np.where(x['observation_count'] >=1, 1, 0)
# predicts
x = x.merge(predicts[['id', 'target']])

x = x.query('observation_count == 0').sort_values('target', ascending=False).head(budget)
grid_gdf_best = grid_gdf.merge(x, on='id', how='inner')

grid_gdf_best = grid_gdf_best.to_crs("EPSG:4326")
grid_gdf_best['longitude'] = grid_gdf_best.geometry.centroid.x
grid_gdf_best['latitude'] = grid_gdf_best.geometry.centroid.y
# --------------

# Plotting cities
fig_plotly = plots.plot_city(df_city, grid_gdf, predicts, 'target', grid_gdf_best)



# Create two columns (panels)
left_col, right_col = st.columns([0.7, 1.3])

# Right Panel Content
with right_col:
    st.header("Mapa miasta")

    # Response to clicking on map
    clicked_points = plotly_events(
        fig_plotly,
        click_event=True,
        select_event=False,
        hover_event=False,
        override_height=700,
        override_width="100%",
    )

    # Show clicked info
    if clicked_points:
        clicked_id = clicked_points[0].get("pointIndex")
    else:
        clicked_id = 1


# Left Panel Content
with left_col:
    # st.header("Informacje o predykcji")
    # preds = models.explain_prediction(grid_gdf, clicked_id, model, X_train, y_train)
    # preds.result.iloc[len(preds.result) -1, 0 ] = 'total'
    
    # # Create the figure
    # fig, ax = plt.subplots(1, 1, figsize=(10, 30))

    # # Plot your data
    # ax.plot(preds.result['cumulative'], preds.result['variable_name'], '-')

    # # Show in Streamlit
    # st.pyplot(fig)

    # # Info text about clicked-on segment
    # st.write(f"Statystyki wyświetlane dla segmentu: {clicked_id}")
    st.header("Informacje o predykcji")
    fig = models.explain_prediction(grid_gdf, clicked_id, model, X_train, y_train)
    st.pyplot(fig)
    st.write(f"Statystyki wyświetlane dla segmentu: {clicked_id}")
    #fig_html = models.explain_prediction_force_plot(grid_gdf, clicked_id, model, X_train, y_train)
    #components.html(fig_html, height=400)
