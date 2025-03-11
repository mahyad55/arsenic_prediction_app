import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

# Sample Data: Latitude, Longitude, and Value
# data = pd.DataFrame({
#     "lat": [35.6892, 34.0522, 40.7128, 48.8566, 51.5074],  # Sample latitudes
#     "lon": [51.3890, -118.2437, -74.0060, 2.3522, -0.1278],  # Sample longitudes
#     "value": [10, 50, 80, 30, 100]  # Sample values
# })

data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=["lat", "lon"],
)
data['value'] = np.random.rand(1000, 1) * 100
# Define Color Scaling: Red (Low) → Green (High)
def get_color(value):
    """Map value to a red-to-green gradient."""
    red = int(255 * (1 - value / 100))  # More value → Less Red
    green = int(255 * (value / 100))    # More value → More Green
    return [red, green, 0, 160]  # RGBA (last is transparency)

data["color"] = data["value"].apply(get_color)  # Apply color mapping
st.dataframe(data.head())

# Pydeck Layer for Interactive Map
layer = pdk.Layer(
    "ScatterplotLayer",
    data,
    get_position=["lon", "lat"],
    get_color="color",
    get_radius=100,  # Adjust radius size
    pickable=True  # Enable click events
)

# View Configuration
view_state = pdk.ViewState(
    latitude=data["lat"].mean(),  # Center map at mean latitude
    longitude=data["lon"].mean(),  # Center map at mean longitude
    zoom=2  # Adjust zoom level
)

# Render Map
map_deck = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={"html": "<b>Value:</b> {value}<br><b>Lat:</b> {lat}<br><b>Lon:</b> {lon}"}
)

# Display the map
st.pydeck_chart(map_deck)

# Handle Click Interaction: Display clicked data below the map
# Get the clicked point data through pickable functionality

st.write("Click on a point to see details.")
