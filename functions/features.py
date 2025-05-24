import pandas as pd

import polars as pl
import polars as pl
import numpy as np

import geopandas

def add_hex_features(df_city, grid_gdf, grid_with_counts, city_data):
    city_data = city_data.select(
        pl.col('centroid_lon', 'centroid_lat', 'type')
    ).with_columns(
            pl.col('type').fill_null('unknown').cast(pl.Categorical)
    )
    observations_gdf = geopandas.GeoDataFrame(
    {'geometry': geopandas.points_from_xy(city_data['centroid_lon'], city_data['centroid_lat'])}, crs="EPSG:4326")
    observations_gdf = pd.concat(
    [observations_gdf.to_crs("EPSG:2180"), city_data['type'].to_pandas()], axis=1
)
    points_in_grid = geopandas.sjoin(observations_gdf, grid_gdf, how="inner", predicate="within")
    observations_per_cell = points_in_grid.groupby(["id", 'type'], group_keys=False, observed=False).size().rename("observation_count").reset_index()

    grid_features = grid_gdf.merge(observations_per_cell, on='id', how="left")
    grid_features['observation_count'] = grid_features['observation_count'].fillna(0)
    grid_features = grid_features.drop(columns='geometry').set_index(['id', 'type']).unstack(-1)
    grid_features.columns = ['_'.join([str(j) for j in i])for i in grid_features.columns]
    grid_features = grid_features.drop(columns=['observation_count_nan'])
    grid_features = grid_features.fillna(0)
    grid_gdf = grid_gdf.merge(grid_features, on='id')
    grid_with_counts = grid_with_counts.merge(grid_features, on='id')
    return (grid_gdf, grid_with_counts)


