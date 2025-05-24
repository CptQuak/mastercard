import matplotlib.pyplot as plt
import dalex as dx 

import numpy as np
import pandas as pd

import polars as pl
import seaborn as sns
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
import polars as pl
import plotly.express as px
import plotly.graph_objects as go 
import numpy as np
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression

import geopandas
import plotly.express as px
from shapely.geometry import Polygon, box
import numpy as np 
from geopy.geocoders import Nominatim
from sklearn.pipeline import make_pipeline


def _add_features(grid_gdf, train=True, model=None):
    if train:
        X, y = pd.DataFrame(grid_gdf.drop(columns=['id','geometry', 'target'])), grid_gdf['target']
    else:
        X = pd.DataFrame(grid_gdf.drop(columns=['id','geometry']))
        for i in X.columns:
            if i not in model.feature_names_in_:
                X = X.drop(columns=[i])
        for i in model.feature_names_in_:
            if i not in X.columns:
                X[i] = 0
        X = X[model.feature_names_in_]
    if train:
        return X, y
    else:
        return X


def _create_training_frame(df_city, grid_gdf, grid_with_counts):
    train_df = grid_with_counts.copy()
    train_df['target'] = np.where(train_df['observation_count'] >=1, 1, 0)
    train_df = train_df.drop(columns=['observation_count'])
    X, y = _add_features(train_df)
    return X, y


def model_fit(df_city, grid_gdf, grid_with_counts):
    X, y = _create_training_frame(df_city, grid_gdf, grid_with_counts)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(fit_intercept=False, max_iter=10_000)
    ).fit(X_train, y_train)
    return model, X, y
    
    
def model_predict(model, df_city, grid_gdf, grid_with_counts):
    X_eval = _add_features(grid_gdf, train=False, model=model)
    predicts = grid_with_counts.copy()
    predicts['target'] = model.predict_proba(X_eval)[:, 1]
    return predicts



# def explain_prediction(grid_gdf, idx, model, X_train, y_train):
#     X_eval = _add_features(grid_gdf, train=False, model=model)
#     exp = dx.Explainer(model, X_train, y_train)
#     obs = pd.DataFrame(X_eval.iloc[[idx], :])
#     return exp.predict_parts(obs) 

def explain_prediction(grid_gdf, idx, model, X_train, y_train):
    X_eval = _add_features(grid_gdf, train=False, model=model)
    import shap

    X_eval_single = X_eval.iloc[idx:]
    explainer = shap.Explainer(model.predict, X_eval_single)
    shap_values = explainer(X_eval_single)
    fig = shap.plots.waterfall(shap_values[1])
    return fig

def explain_prediction_force_plot(grid_gdf, idx, model, X_train, y_train):
    X_eval = _add_features(grid_gdf, train=False, model=model)
    import shap

    X_eval_single = X_eval.iloc[idx:]
    explainer = shap.Explainer(model.predict, X_eval_single)
    shap_values = explainer(X_eval_single)

    force_plot_html = shap.force_plot(
        base_value=shap_values.base_values[0], 
        shap_values=shap_values.values[0], 
        features=X_eval_single.iloc[0], 
        feature_names=X_eval_single.columns
    ).data
    return force_plot_html