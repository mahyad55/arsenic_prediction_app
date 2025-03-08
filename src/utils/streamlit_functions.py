# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 11:57:38 2025

@author: MahYad
"""

import plotly.express as px
import numpy as np
import folium
from folium.plugins import HeatMap
from branca.colormap import linear

# %% define clasa
class StreamLit:
    
    # Function to generate a color range from green to red
    def get_color(self, value):
        # Define the color range from green to red
        green = np.array([0, 255, 0])  # RGB for green
        red = np.array([255, 0, 0])  # RGB for red
    
        # Interpolate between green and red based on the value
        color = green + (red - green) * value
        return tuple(color.astype(int))  # Convert to tuple of integers for color
    
    
    # Function to create 10 color steps from green to red
    def generate_color_range(self):
        return [self.get_color(i / 10) for i in range(11)]
    
    def select_color(self, pred_proba, color_range):
        if pred_proba<0.1:
            color = color_range[0]
        elif pred_proba<0.2:
            color = color_range[1]
        elif pred_proba<0.3:
            color = color_range[2]
        elif pred_proba<0.4:
            color = color_range[3]
        elif pred_proba<0.5:
            color = color_range[4]
        elif pred_proba<0.6:
            color = color_range[5]
        elif pred_proba<0.7:
            color = color_range[6]
        elif pred_proba<0.8:
            color = color_range[7]
        elif pred_proba<0.9:
            color = color_range[8]
        elif pred_proba<1:
            color = color_range[9]
            
        return color
    
    def rgb_to_hex(self, rgb):
        return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])
    
    
    def map_plot(self, y_huc_results):
        fig_map = px.scatter_mapbox(
                y_huc_results.reset_index(),
                lat='LAT',
                lon='LONG',
                color='pred_proba',
                hover_data=['HUC_12', 'LAT', 'LONG', 'pred_proba'],
                color_continuous_scale="RdYlGn_r",  # Reversed so red is high and green is low
                size_max=15,
                mapbox_style="carto-positron",  # Choose a map style
    
                zoom=3.2,
                height=600,
                width=800
            )
    
        fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        fig_map.update_layout(mapbox_style="open-street-map")
        fig_map.update_layout(mapbox_accesstoken='your_mapbox_token')
        fig_map.update_traces(marker=dict(size=7))  # Set to any fixed value
        
        return fig_map
    
    def map_plot_folium(df):
        
        # Create a folium map centered at an average location
        map_center = [df['LAT'].mean(), df['long'].mean()]
        heatmap = folium.Map(location=map_center, zoom_start=6, control_scale=True)
        
        # Create a color scale from green to red using predproba values
        colormap = linear.RdYlGn_r.scale(df['pred_proba'].min(), df['pred_proba'].max())
        
        # Add a HeatMap layer
        heat_data = [[row['LAT'], row['LONG'], row['pred_proba']] for _, row in df.iterrows()]
        HeatMap(
            heat_data, min_opacity=0.5, radius=15, blur=10,
            gradient={0.4: 'green', 0.6: 'yellow', 1: 'red'}
        ).add_to(heatmap)
        
        # Add markers for each point with a popup showing all information
        for _, row in df.iterrows():
            folium.CircleMarker(
                location=[row['LAT'], row['long']],
                radius=8,
                color=colormap(row['pred_proba']),
                fill=True,
                fill_color=colormap(row['pred_proba']),
                fill_opacity=0.7
            ).add_to(heatmap)
        
            # Add a popup with HUC, LAT, LONG, and predproba
            folium.Popup(
                f"HUC: {row['HUC_12']}<br>LAT: {row['LAT']}<br>LON: {row['long']}<br>pred_proba: {row['pred_proba']}"
            ).add_to(heatmap)
        
        # Add the colormap to the map
        colormap.add_to(heatmap)
        
        # Display the map in Streamlit (or save to HTML file)
        heatmap.save("heatmap_with_popup.html")
        return heatmap
    
    def selected_map(self, lat, long, pred_proba):
        
        color_range = self.generate_color_range()
        
        mymap = folium.Map(location=[lat, long], zoom_start=12)
        color = self.select_color(pred_proba, color_range)
        color = self.rgb_to_hex(color)
        
        folium.Circle(
            location=[lat, long],
            radius=500,  # radius in meters
            color='red',  # Border color of the circle
            fill=True,  # Fill the circle
            fill_color=color,  # Fill color inside the circle
            fill_opacity=0.4  # Opacity of the circle fill
        ).add_to(mymap)
        
        return mymap
