import numpy as np

import polars as pl
import polars as pl
import numpy as np

import geopandas
from shapely.geometry import Polygon, box
import numpy as np 
from geopy.geocoders import Nominatim


def get_city_bounds(city_name, country='Poland'):
    """Get city box min max lat lon"""
    geolocator = Nominatim(user_agent="city_bounds_script")
    location = geolocator.geocode(f"{city_name}, {country}", exactly_one=True)
    if location and hasattr(location, 'raw') and 'boundingbox' in location.raw:
        bounding_box = location.raw['boundingbox']
        min_lat, max_lat = float(bounding_box[0]), float(bounding_box[1])
        min_lon, max_lon = float(bounding_box[2]), float(bounding_box[3])
        return {
            'city': city_name,
            'min_lat': min_lat,
            'max_lat': max_lat,
            'min_lon': min_lon,
            'max_lon': max_lon
        }
    else:
        raise Exception(str({'city': city_name, 'error': 'City not found or no bounding box'}))


def generate_boxes(min_lon, min_lat, max_lon, max_lat, cell_size):
    """Generate boxes based on cooridnates"""
    aoi_bbox_gdf = geopandas.GeoDataFrame(
        {'geometry': [box(min_lon, min_lat, max_lon, max_lat)]}, crs="EPSG:4326")
    aoi_bbox_projected = aoi_bbox_gdf.to_crs("EPSG:2180")
    minx, miny, maxx, maxy = aoi_bbox_projected.total_bounds
    grid_cells_data = []
    cell_id_counter = 0
    for x0 in np.arange(minx, maxx, cell_size):
        for y0 in np.arange(miny, maxy, cell_size):
            x1 = x0 + cell_size; y1 = y0 + cell_size
            grid_cells_data.append({'id': f'cell_{cell_id_counter}', 'geometry': Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])})
            cell_id_counter += 1
    grid_gdf = geopandas.GeoDataFrame(grid_cells_data, crs="EPSG:2180")
    return grid_gdf

def create_city_grid(df, city:str = 'Białystok', cell_size:int = 250):
    """Create grid for a specified city"""
    df_city = df.filter(pl.col('line2').str.contains(city))
    boundaries = get_city_bounds(city)
    
    lower_bound_lat = boundaries['min_lat']
    upper_bound_lat = boundaries['max_lat']
    lower_bound_lon = boundaries['min_lon']
    upper_bound_lon = boundaries['max_lon']
    

    df_city = df_city.filter(
        (pl.col('latitude') >= lower_bound_lat) &
        (pl.col('latitude') <= upper_bound_lat) &
        (pl.col('longitude') >= lower_bound_lon) &
        (pl.col('longitude') <= upper_bound_lon)
    )
    min_lon, min_lat = df_city['longitude'].min(), df_city['latitude'].min()
    max_lon, max_lat = df_city['longitude'].max(), df_city['latitude'].max()
    
    print(min_lon, min_lat)
    print(max_lon, max_lat)
    # generate boxes on map
    grid_gdf = generate_boxes(min_lon, min_lat, max_lon, max_lat, cell_size)
    
    # merge boxes with original frame, mark non matches with 0
    observations_gdf = geopandas.GeoDataFrame(
        {'geometry': geopandas.points_from_xy(df['longitude'], df['latitude'])}, crs="EPSG:4326")
    observations_gdf = observations_gdf.to_crs("EPSG:2180")

    points_in_grid = geopandas.sjoin(observations_gdf, grid_gdf, how="inner", predicate="within")
    observations_per_cell = points_in_grid.groupby("id").size().rename("observation_count")
    grid_with_counts = grid_gdf.merge(observations_per_cell, left_on="id", right_index=True, how="left").fillna(0)
    return df_city, grid_gdf, grid_with_counts
