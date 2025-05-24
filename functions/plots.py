import plotly.express as px
import plotly.graph_objects as go 

def plot_city(df_city, grid_gdf, grid_with_counts, value_column='observation_count'):
    map_center_lat = df_city["latitude"].mean()
    map_center_lon = df_city["longitude"].mean()

    # --- Create the Choropleth Map (as the base) ---
    grid_with_counts_4326 = grid_with_counts.to_crs(epsg=4326)
    df_for_plotly = grid_with_counts_4326.set_index('id')

    fig = px.choropleth_mapbox( # Use choropleth_mapbox for compatibility
        df_for_plotly,
        geojson=df_for_plotly.geometry.__geo_interface__, # Use __geo_interface__
        locations=df_for_plotly.index,
        color=value_column,
        color_continuous_scale="Viridis",
        range_color=(0, df_for_plotly[value_column].max()),
        mapbox_style="carto-positron", # Choose a mapbox style
        zoom=10,
        center={"lat": map_center_lat, "lon": map_center_lon},
        opacity=0.5, # Adjusted opacity to see scatter points
        labels={value_column: 'Observations'},
        hover_data={value_column: True}
    )

    # --- Create the Scatter Map traces ---
    # We will create a scatter_mapbox trace and add it
    scatter_trace = go.Scattermapbox(
        lat=df_city["latitude"].to_list(),
        lon=df_city["longitude"].to_list(),
        mode='markers+text',
        marker=go.scattermapbox.Marker(
            size=6,
            color='blue',
            opacity=0.7
        ),
        text=["📦"] * len(df_city),
        textposition="top right",
        textfont=dict(size=14),
        name="Paczkomaty"
    )

    fig.add_trace(scatter_trace)

    # --- Update Layout ---
    fig.update_layout(
        margin={"r":0,"t":40,"l":0,"b":0},
        title_text="Interakcyjna mapa z obserwacjami",
        title_x=0.5,
        height=700,
        legend_title_text='Layers'
    )

    return fig