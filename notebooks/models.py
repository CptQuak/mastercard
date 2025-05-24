import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import polars as pl
import seaborn as sns
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


def add_features(grid_gdf, train=True):
    if train:
        X, y = np.ones((len(grid_gdf), 1)), grid_gdf['target']
    else:
        X =  np.ones((len(grid_gdf), 1))
    if train:
        return X, y
    else:
        return X


def create_training_frame(df_city, grid_gdf, grid_with_counts):
    train_df = grid_with_counts.copy()
    train_df['target'] = np.where(train_df['observation_count'] >=1, 1, 0)
    train_df = train_df.drop(columns=['observation_count'])
    X, y = add_features(train_df)
    return X, y


def model_fit(df_city, grid_gdf, grid_with_counts):
    X, y = create_training_frame(df_city, grid_gdf, grid_with_counts)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    model = LogisticRegressionCV().fit(X_train, y_train)
    return model
    
    
def model_predict(model, df_city, grid_gdf, grid_with_counts):
    X_eval = add_features(grid_gdf, train=False)
    predicts = grid_with_counts.copy()
    predicts['target'] = model.predict_proba(X_eval)[:, 1]
    return predicts