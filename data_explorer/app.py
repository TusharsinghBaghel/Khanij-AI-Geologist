import geopandas as gpd
import folium
import numpy as np
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
import io

from io import BytesIO
import base64
import json
import pandas as pd
from folium.plugins import HeatMapWithTime
from scipy.interpolate import griddata,bisplrep, bisplev
from pykrige.ok import OrdinaryKriging
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mayavi import mlab
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy import stats

import gradio as gr


from groq import Groq
client = Groq(
    api_key="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",
)




def create_heatmap(geojson_file, chem_symbol, lat_min, lat_max, long_min, long_max):
    """
    Generates a heatmap from geochemical data within specified latitudinal and longitudinal limits.

    Parameters:
    - geojson_file (str): The file path to the GeoJSON file containing geochemical data.
    - chem_symbol (str): The chemical symbol to visualize.
    - lat_min (float): The minimum latitude value.
    - lat_max (float): The maximum latitude value.
    - long_min (float): The minimum longitude value.
    - long_max (float): The maximum longitude value.

    Returns:
    - dict: A dictionary containing the message and HTML content of the generated heatmap.
            {
                "message": str,
                "html": str (optional)
            }
    """
    # Define lat/long limits
    lat_limits = (lat_min, lat_max)
    long_limits = (long_min, long_max)

    try:
        # Load GeoJSON data
        gdf = gpd.read_file(geojson_file)
        
        # Check if the chemical symbol exists in the data
        if chem_symbol not in gdf.columns:
            return {"message": f"Error: Chemical symbol '{chem_symbol}' not found in the data."}
        
        # Filter data for the specified lat/long limits
        gdf = gdf[(gdf.geometry.y >= lat_limits[0]) & (gdf.geometry.y <= lat_limits[1]) & 
                  (gdf.geometry.x >= long_limits[0]) & (gdf.geometry.x <= long_limits[1])]
        
        if gdf.empty:
            return {"message": "Error: No data available within the specified lat/long limits."}
        
        # Prepare data for heatmap
        heat_data = [[point.y, point.x, value] for point, value in zip(gdf.geometry, gdf[chem_symbol])]
        
        # Base map centered on the midpoint of the given lat/long limits
        lat_center = (lat_limits[0] + lat_limits[1]) / 2
        long_center = (long_limits[0] + long_limits[1]) / 2
        m = folium.Map(location=[lat_center, long_center], zoom_start=10)
        
        # Add heatmap layer
        heatmap_layer = HeatMap(heat_data, min_opacity=0.2, radius=15, blur=15)
        heatmap_layer.add_to(m)
        
        # Add invisible markers with tooltips for interactivity
        for point, value in zip(gdf.geometry, gdf[chem_symbol]):
            folium.Marker(
                location=[point.y, point.x],
                icon=folium.DivIcon(html=f"""<div style="display:none;">{value}</div>"""),
                tooltip=f'{chem_symbol}: {value}'
            ).add_to(m)
        
        # Add layer control to toggle heatmap
        folium.LayerControl().add_to(m)
        
        # Generate the HTML content
        html_content = m.get_root().render()
        
        return {
            "message": f"Heatmap for '{chem_symbol}' generated successfully.",
            "html": html_content
        }
    
    except Exception as e:
        return {"message": f"An error occurred: {str(e)}"}


def create_geochem_avg_histogram(geojson_file, chem_symbols, lat_min, lat_max, long_min, long_max):
    
    try:
        # Load the GeoJSON data
        geo_data = gpd.read_file(geojson_file)
        
        # Filter the data by latitude and longitude bounds
        filtered_data = geo_data.cx[long_min:long_max, lat_min:lat_max]
        # Initialize a dictionary to store average values for each chemical
        avg_values = {}
        
        # Calculate average concentration for each chemical symbol
        for chem_symbol in chem_symbols:
            if chem_symbol in filtered_data.columns:
                avg_value = filtered_data[chem_symbol].mean()
                avg_values[chem_symbol] = avg_value
            # If there's no data for the chemical symbol, skip it
            else:
                print(f"no data for {chem_symbol}")
            
        # Find the chemical with the highest average concentration
        max_avg_chem = max(avg_values, key=avg_values.get) if avg_values else None
        max_avg_value = avg_values[max_avg_chem] if max_avg_chem else None
        
        # Create the histogram
        if avg_values:
            plt.figure(figsize=(10, 6))
            plt.bar(avg_values.keys(), avg_values.values(), color='skyblue')
            plt.xlabel('Chemical Symbols')
            plt.ylabel('Average Concentration')
            plt.title('Average Concentration of Chemicals')
            
            # Save the plot to a BytesIO object
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # Convert the plot to a base64 string
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            # Create the HTML content
            html_content = f'<img src="data:image/png;base64,{img_base64}" />'
        else:
            html_content = '<p>No data available for the specified chemicals and area.</p>'
        
        # Create the JSON response
        response = {
            "message": f"Take a look at the histogram with maximum being {max_avg_chem} with value of {max_avg_value}.",
            "html": html_content
        }
        
        return response
    except Exception as e:
        return {"message": f"An error occurred: {str(e)}"}


def interpolation_geojson(json_file, chem_symbol, lat_min, lat_max, long_min, long_max, method):
    try:
        # Load geochemical data from JSON file into a DataFrame
        with open(json_file, 'r') as f:
            data = json.load(f)

        features = data['features']

        # Extract relevant properties into lists
        coordinates = []
        values = []

        for feature in features:
            properties = feature['properties']
            geometry = feature['geometry']
            latitude = geometry['coordinates'][1]
            longitude = geometry['coordinates'][0]
            if lat_min <= latitude <= lat_max and long_min <= longitude <= long_max:
                value = properties.get(chem_symbol, None)
                if value is not None and not np.isnan(value):  # Check for valid values
                    coordinates.append([latitude, longitude])
                    values.append(value)

        if not coordinates:
            return {"message":"No data found within specified coordinates."}

        # Create a DataFrame from extracted data
        df = pd.DataFrame({'Latitude': [coord[0] for coord in coordinates],
                           'Longitude': [coord[1] for coord in coordinates],
                           'Value': values})

        # Prepare grid for interpolation
        grid_lat = np.linspace(lat_min, lat_max, 100)
        grid_long = np.linspace(long_min, long_max, 100)
        grid_lat, grid_long = np.meshgrid(grid_lat, grid_long)

        # Perform IDW interpolation
        points = np.array(coordinates)
        values = np.array(values)
        interpolation_type="nan"
        if method==1:
            #IDW Interpolation
            interpolation_type = "IDW"
            grid_z = griddata(points, values, (grid_lat, grid_long), method='cubic')
        elif method==2:
            #Spline Interpolation:
            interpolation_type = "Spline"
            tck = bisplrep(points[:, 0], points[:, 1], values, s=0)
            #s is the smoothing value and is chosen by trial and error
            grid_z = bisplev(grid_lat[:, 0], grid_long[0, :], tck)
        elif method==3:
            #Kriging interpolation
            interpolation_type = "Kriging"
            OK = OrdinaryKriging(points[:, 0], points[:, 1], values, variogram_model='linear')
            grid_z, _ = OK.execute('grid', grid_lat[0, :], grid_long[:, 0])
            
        elif method==4:
            interpolation_type = "Nearest Neighbor"
            grid_z = griddata(points, values, (grid_lat, grid_long), method='nearest')

        # Initialize the map centered around the mean of latitude and longitude limits
        map_center = [(lat_min + lat_max) / 2, (long_min + long_max) / 2]
        my_map = folium.Map(location=map_center, zoom_start=10)

        # Add heatmap layer to the map
        heat_data = []
        for i in range(grid_lat.shape[0]):
            for j in range(grid_long.shape[1]):
                if not np.isnan(grid_z[i, j]):
                    heat_data.append([grid_lat[i, j], grid_long[i, j], grid_z[i, j]])

        HeatMap(heat_data, radius=15, blur=25, max_zoom=18).add_to(my_map)

        # Add legend to the map
        caption = f'{interpolation_type} Interpolated {chem_symbol} Heatmap'
        my_map.get_root().html.add_child(folium.Element(f'<div style="position: fixed; bottom: 50px; left: 50px; z-index:9999; background-color:white; padding: 10px; border: 2px solid grey; border-radius: 5px;">{caption}</div>'))

        # Save the map as an HTML string
        map_html = my_map.get_root().render()

        return {"message":f"Heatmap of {interpolation_type} interpolated data created successfully for {chem_symbol}.","html": map_html}

    except Exception as e:
        return {"message":f"An error occurred: {str(e)}"}

def plot_excavation_sites(geojson_file, lat_min, lat_max, long_min, long_max):
    """
    Plots excavation sites from GeoJSON data as markers on a map with hover tooltips and generates HTML content.

    Parameters:
    - geojson_file (str): The file path to the GeoJSON file containing excavation site data.
    - lat_min (float): The minimum latitude value.
    - lat_max (float): The maximum latitude value.
    - long_min (float): The minimum longitude value.
    - long_max (float): The maximum longitude value.

    Returns:
    - dict: A dictionary containing the message and HTML content of the generated map.
            {
                "message": str,
                "html": str (optional)
            }
    """
    # Initialize the map centered around the midpoint of the specified lat/long limits
    lat_center = (lat_min + lat_max) / 2
    long_center = (long_min + long_max) / 2
    m = folium.Map(location=[lat_center, long_center], zoom_start=10)

    try:
        # Load GeoJSON data
        with open(geojson_file, 'r') as f:
            data = json.load(f)

        # Iterate through each feature in the GeoJSON file
        for feature in data['features']:
            properties = feature['properties']
            geometry = feature['geometry']

            if geometry['type'] == 'Polygon':
                coordinates = geometry['coordinates'][0]  # Extracting the coordinates of the Polygon

                # Check if any point of the polygon falls within the specified lat/long limits
                for coord in coordinates:
                    if (lat_min <= coord[1] <= lat_max and
                        long_min <= coord[0] <= long_max):
                        #"period_of_propecting_from": "1988", "period_of_propecting_to": "1989"
                        # Add marker for each excavation site
                        marker = folium.Marker(location=[coord[1], coord[0]],
                                               tooltip=f"Commodity: {properties.get('commodity', 'N/A')}",
                                               popup=f"<strong>Commodity:</strong> {properties.get('commodity', 'N/A')}<br>"
                                                     f"<strong>Exploration Agency:</strong> {properties.get('name_of_exploration_agency', 'N/A')}<br>"
                                                     f"<strong>Project Title:</strong> {properties.get('project_title', 'N/A')}<br>"
                                                     f"<strong>Exploration Stage:</strong> {properties.get('exploration_stage', 'N/A')}<br>"
                                                     f"<strong>from:</strong> {properties.get('period_of_propecting_from', 'N/A')}<strong> to:</strong>{properties.get('period_of_propecting_to', 'N/A')}<br>"
                                                     f"<strong>Prospector name:</strong> {properties.get('name_of_the_prospector','N/A')}<br>")
                        marker.add_to(m)

        # Generate the HTML content
        html_content = m.get_root().render()

        return {
            "message": "Excavation sites plotted successfully.",
            "html": html_content
        }

    except Exception as e:
        return {"message": f"An error occurred: {str(e)}"}



def create_dem_mayavi( geojson_file, lat_min, lat_max, long_min, long_max):
    """
    Creates a DEM from geomorphological data in a GeoJSON file and returns an HTML representation.

    Parameters:
    - lat_min, lat_max: Latitude boundaries.
    - long_min, long_max: Longitude boundaries.
    - geojson_file: Path to the GeoJSON file.

    Returns:
    - dict: A dictionary with a message and HTML content of the DEM.
    """
    try:
        # Load GeoJSON data
        gdf = gpd.read_file(geojson_file)
        
        # Filter data within specified lat/long limits
        gdf = gdf.cx[long_min:long_max, lat_min:lat_max]
        
        if gdf.empty:
            return {"message": "Error: No data available within the specified lat/long limits."}
        
        # Extract coordinates and elevation values
        coordinates = []
        elevations = []
        
        for feature in gdf.itertuples():
            if feature.geometry.type == 'Polygon':
                coords = np.array(feature.geometry.exterior.coords)
                coordinates.extend(coords)
                elevations.extend([feature.shape_leng] * len(coords))
            # elif feature.geometry.type == 'MultiPolygon':
            #     for polygon in feature.geometry:
            #         coords = np.array(polygon.exterior.coords)
            #         coordinates.extend(coords)
            #         elevations.extend([feature.shape_leng] * len(coords))
        
        coordinates = np.array(coordinates)
        elevations = np.array(elevations)
        
        # Generate grid for DEM
        lats = np.linspace(lat_min, lat_max, 100)
        longs = np.linspace(long_min, long_max, 100)
        longs, lats = np.meshgrid(longs, lats)
        
        # Interpolate elevation values onto grid
        grid_z = griddata(coordinates[:, :2], elevations, (longs, lats), method='cubic')
        
        # Create DEM plot
        fig = mlab.figure(size=(800, 600), bgcolor=(1, 1, 1))
        dem_plot = mlab.surf(longs, lats, grid_z, warp_scale='auto', colormap='terrain')
        mlab.colorbar(title='Elevation', orientation='vertical')
        
        # Save the plot as HTML
        html_output = 'dem_plot.html'
        mlab.savefig(html_output)
        mlab.close()
        
        with open(html_output, 'r') as file:
            html_content = file.read()
        
        return {"message": "DEM generated successfully.", "html": html_content}
    
    except Exception as e:
        return {"message": f"An error occurred: {str(e)}"}


def create_dem(geojson_file, lat_min, lat_max, long_min, long_max):
    """
    Creates a 3D DEM from geomorphological data in a GeoJSON file and returns an HTML representation.

    Parameters:
    - lat_min, lat_max: Latitude boundaries.
    - long_min, long_max: Longitude boundaries.
    - geojson_file: Path to the GeoJSON file.

    Returns:
    - dict: A dictionary with a message and HTML content of the DEM.
    """
    try:
        # Load GeoJSON data
        gdf = gpd.read_file(geojson_file)
        
        # Filter data within specified lat/long limits
        gdf = gdf.cx[long_min:long_max, lat_min:lat_max]
        
        if gdf.empty:
            return {"message": "Error: No data available within the specified lat/long limits."}
        
        # Extract coordinates and elevation values
        coordinates = []
        elevations = []
        
        for feature in gdf.itertuples():
            if feature.geometry.geom_type == 'Polygon':
                coords = np.array(feature.geometry.exterior.coords)
                coordinates.extend(coords)
                elevations.extend([feature.shape_leng] * len(coords))
            # elif feature.geometry.geom_type == 'MultiPolygon':
            #     for polygon in feature.geometry:
            #         coords = np.array(polygon.exterior.coords)
            #         coordinates.extend(coords)
            #         elevations.extend([feature.shape_leng] * len(coords))
        
        coordinates = np.array(coordinates)
        elevations = np.array(elevations)
        
        # Generate grid for DEM
        lats = np.linspace(lat_min, lat_max, 100)
        longs = np.linspace(long_min, long_max, 100)
        longs, lats = np.meshgrid(longs, lats)
        
        # Interpolate elevation values onto grid
        grid_z = griddata(coordinates[:, :2], elevations, (longs, lats), method='cubic')
        
        # Create 3D DEM plot using matplotlib
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(longs, lats, grid_z, cmap='terrain')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('Elevation')
        ax.set_title('Digital Elevation Model (DEM)')
        max_elevation = np.nanmax(grid_z)
        ax.set_zlim(0, max_elevation)
        # Save the plot to a bytes buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        
        # Encode the plot to a base64 string
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode('utf-8')
        html_content = f'<img src="data:image/png;base64,{img_str}" />'
        
        return {"message": "3D DEM generated successfully.", "html": html_content}
    
    except Exception as e:
        return {"message": f"An error occurred: {str(e)}"}

from shapely.geometry import LineString

def create_contour_map(geojson_file, chem_symbol, lat_min, lat_max, long_min, long_max):
    """
    Generates a customized contour map overlay on a Folium map from geochemical data within specified latitudinal and longitudinal limits.

    Parameters:
    - geojson_file (str): The file path to the GeoJSON file containing geochemical data.
    - chem_symbol (str): The chemical symbol to visualize.
    - lat_min (float): The minimum latitude value.
    - lat_max (float): The maximum latitude value.
    - long_min (float): The minimum longitude value.
    - long_max (float): The maximum longitude value.

    Returns:
    - dict: A dictionary containing the message and HTML content of the generated contour map overlay.
            {
                "message": str,
                "html": str (optional)
            }
    """
    try:
        # Load GeoJSON data
        gdf = gpd.read_file(geojson_file)
        
        # Check if the chemical symbol exists in the data
        if chem_symbol not in gdf.columns:
            return {"message": f"Error: Chemical symbol '{chem_symbol}' not found in the data."}
        
        # Filter data for the specified lat/long limits
        gdf = gdf[(gdf.geometry.y >= lat_min) & (gdf.geometry.y <= lat_max) & 
                  (gdf.geometry.x >= long_min) & (gdf.geometry.x <= long_max)]
        
        if gdf.empty:
            return {"message": "Error: No data available within the specified lat/long limits."}
        
        # Prepare data for contour plot
        latitudes = gdf.geometry.y.values
        longitudes = gdf.geometry.x.values
        values = gdf[chem_symbol].values
        
        # Create grid data
        grid_lat, grid_long = np.mgrid[lat_min:lat_max:100j, long_min:long_max:100j]
        grid_values = griddata((latitudes, longitudes), values, (grid_lat, grid_long), method='cubic')
        
        # Generate the contour plot with custom colormap
        fig, ax = plt.subplots()
        contour = ax.contourf(grid_long, grid_lat, grid_values, cmap=cm.jet, alpha=0.6)  # Adjust the colormap (cm.spring is just an example)
        cbar = plt.colorbar(contour, ax=ax, label=chem_symbol)
        
        # Customize plot appearance
        ax.axis('off')  # Turn off axis lines
        
        # Save the contour plot to a PNG image in memory
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        # Base map centered on the midpoint of the given lat/long limits
        lat_center = (lat_min + lat_max) / 2
        long_center = (long_min + long_max) / 2
        m = folium.Map(location=[lat_center, long_center], zoom_start=12)
        
        # Overlay the contour plot image on the map
        folium.raster_layers.ImageOverlay(
            image=f'data:image/png;base64,{image_base64}',
            bounds=[[lat_min, long_min], [lat_max, long_max]],
            opacity=0.6
        ).add_to(m)
        
        # Add layer control to toggle contour image
        folium.LayerControl().add_to(m)
        
        # Generate the HTML content
        html_content = m.get_root().render()
        
        return {
            "message": f"Contour map for '{chem_symbol}' generated successfully.",
            "html": html_content
        }
    
    except Exception as e:
        return {"message": f"An error occurred: {str(e)}"}
                    

def plot_high_value_points(geojson_file, chem_symbol, lat_min, lat_max, long_min, long_max):
    """
    Plots points on a map for geochemical data that have values higher than the average value of the chemical.
    Also calculates and includes statistics (mean, mode, median, std deviation, variance) of the chemical values.

    Parameters:
    - geojson_file (str): The file path to the GeoJSON file containing geochemical data.
    - chem_symbol (str): The chemical symbol to visualize.
    - lat_min (float): The minimum latitude value.
    - lat_max (float): The maximum latitude value.
    - long_min (float): The minimum longitude value.
    - long_max (float): The maximum longitude value.

    Returns:
    - dict: A dictionary containing the message and HTML content of the generated map.
            {
                "message": str,
                "html": str (optional)
            }
    """
    # Define lat/long limits
    lat_limits = (lat_min, lat_max)
    long_limits = (long_min, long_max)

    try:
        # Load GeoJSON data
        gdf = gpd.read_file(geojson_file)
        
        # Check if the chemical symbol exists in the data
        if chem_symbol not in gdf.columns:
            return {"message": f"Error: Chemical symbol '{chem_symbol}' not found in the data."}
        
        # Filter data for the specified lat/long limits
        gdf = gdf[(gdf.geometry.y >= lat_limits[0]) & (gdf.geometry.y <= lat_limits[1]) & 
                  (gdf.geometry.x >= long_limits[0]) & (gdf.geometry.x <= long_limits[1])]
        
        if gdf.empty:
            return {"message": "Error: No data available within the specified lat/long limits."}
        
        # Calculate statistics of the chemical values
        chem_values = gdf[chem_symbol].values
        mean_value = np.mean(chem_values)
        mode_value = stats.mode(chem_values)  # Mode can have multiple values, taking the first one
        median_value = np.median(chem_values)
        std_deviation = np.std(chem_values)
        variance = np.var(chem_values)
        
        # Filter points with values higher than the average
        high_value_points = gdf[gdf[chem_symbol] > mean_value]
        
        # Base map centered on the midpoint of the given lat/long limits
        lat_center = (lat_limits[0] + lat_limits[1]) / 2
        long_center = (long_limits[0] + long_limits[1]) / 2
        m = folium.Map(location=[lat_center, long_center], zoom_start=10)
        
        # Add markers for high value points with tooltips
        for idx, row in high_value_points.iterrows():
            folium.Marker(
                location=[row.geometry.y, row.geometry.x],
                tooltip=f'{chem_symbol}: {row[chem_symbol]}'
            ).add_to(m)
        
        # Add layer control to toggle markers
        folium.LayerControl().add_to(m)
        
        # Generate the HTML content
        html_content = m.get_root().render()
        
        # Construct the message with statistics
        message = f"Plotted points with '{chem_symbol}' values higher than mean ({mean_value:.2f})\n"
        message += f"Mean: {mean_value:.2f}\n"
        message += f"Mode: {mode_value}\n"
        message += f"Median: {median_value:.2f}\n"
        message += f"Standard Deviation: {std_deviation:.2f}\n"
        message += f"Variance: {variance:.2f}"
        
        return {
            "message": message,
            "html": html_content
        }
    
    except Exception as e:
        return {"message": f"An error occurred: {str(e)}"}
    

def Manager_agent(query, lat_min, lat_max, long_min, long_max):
    lat_min = float(lat_min)
    lat_max = float(lat_max)
    long_min = float(long_min)
    long_max = float(long_max)
    chat_completion = client.chat.completions.create(
        #
        # Required parameters
        #
        messages=[
            # Set an optional system message. This sets the behavior of the
            # assistant and can be used to provide specific instructions for
            # how it should behave throughout the conversation.
            {
                "role": "system",
                "content": '''You are Khanij, an AI assistant for MECL (Mineral Exploration and Consultancy Limited). Based on the user query, determine the appropriate task to perform:
                            
                            - Print "Heatmap": If the query is related to creating a heatmap.
                            In the next line, print a Python list with one element: the chemical formula(not name) in lowercase mentioned in the query.
                            - If the query mentions geophysical magnetic data, Empty the list and add 'magnetic_a' to the list.
                            - If it's related to gravity,Empty the list and add 'bouguer_an' to the list.
                            - If it's related to elevation,Empty the list and add 'elevation_' to the list.
                            - Do not print anything else.

                            - Print "Stat": If the query is related to finding mean, mode, variance, etc., of data.
                            In the next line, print a Python list with one element: the chemical formula(not name) in lowercase mentioned in the query.

                            - Print "Contour": If the query is related to creating contour maps.
                            In the next line, print a Python list with one element: the chemical formula(not name) in lowercase mentioned in the query.
                            - If the query mentions geophysical magnetic data,Empty the list and add 'magnetic_a' to the list.
                            - If it's related to gravity,Empty the list and add 'bouguer_an' to the list.
                            - If it's related to elevation,Empty the list and add 'elevation_' to the list.
                            - Do not print anything else.

                            - Print "Exploration": If the query is about known exploration or excavation sites of the region.
                            In the next line, print "['Exploration']"
                            - Print "KrigingInterpolation": If the query is related to Kriging interpolation.
                            In the next line, print a Python list with one element: the chemical formula(not name) in lowercase mentioned in the query.
                            - If the query mentions geophysical magnetic data,Empty the list and add 'magnetic_a' to the list.
                            - If it's related to gravity,Empty the list and add 'bouguer_an' to the list.
                            - If it's related to elevation,Empty the list and add 'elevation_' to the list.
                            - Do not print anything else.

                            - Print "IDWInterpolation": If the query is related to IDW (Inverse Distance Weighting) interpolation.
                            In the next line, print a Python list with one element: the chemical formula(not name) in lowercase mentioned in the query.
                            - If the query mentions geophysical magnetic data,Empty the list and add 'magnetic_a' to the list.
                            - If it's related to gravity,Empty the list and add 'bouguer_an' to the list.
                            - If it's related to elevation,Empty the list and add 'elevation_' to the list.
                            - Do not print anything else.

                            - Print "SplineInterpolation": If the query is related to spline interpolation.
                            In the next line, print a Python list with one element: the chemical formula in lowercase mentioned in the query.
                            - If the query mentions geophysical magnetic data,Empty the list and add 'magnetic_a' to the list.
                            - If it's related to gravity,Empty the list and add 'bouguer_an' to the list.
                            - If it's related to elevation,Empty the list and add 'elevation_' to the list.
                            - Do not print anything else.

                            - Print "NNInterpolation": If the query is related to nearest neighbor interpolation.
                            In the next line, print a Python list with one element: the chemical formula(not the name) in lowercase mentioned in the query.
                            - If the query mentions geophysical magnetic data,Empty the list and add 'magnetic_a' to the list.
                            - If it's related to gravity,Empty the list and add 'bouguer_an' to the list.
                            - If it's related to elevation,Empty the list and add 'elevation_' to the list.
                            - Do not print anything else.

                            - Print "Histogram": If the query is related to creating a histogram.
                            In the next line, print a Python list of chemical formulas in lowercase of the chemicals mentioned in the query.
                            - If there are no chemicals mentioned, print a list containing the string 'all'.

                            - Print a response according to your knowledge: If the query does not relate to any of the specified tasks.
                            In the next line print ["No maps defined"]
                            '''
            },
            # Set a user message for the assistant to respond to.
            {
                "role": "user",
                "content": query,
            }
        ],

        # The language model which will generate the completion.
        model="llama3-70b-8192",

        #
        # Optional parameters
        #

        # Controls randomness: lowering results in less random completions.
        # As the temperature approaches zero, the model will become deterministic
        # and repetitive.
        temperature=0.5,

        # The maximum number of tokens to generate. Requests can use up to
        # 2048 tokens shared between prompt and completion.
        max_tokens=1024,

        # Controls diversity via nucleus sampling: 0.5 means half of all
        # likelihood-weighted options are considered.
        top_p=1,

        # A stop sequence is a predefined or user-specified text string that
        # signals an AI to stop generating content, ensuring its responses
        # remain focused and concise. Examples include punctuation marks and
        # markers like "[end]".
        stop=None,

        # If set, partial message deltas will be sent.
        stream=False,
    )

    #print(chat_completion.choices[0].message.content)

    response = chat_completion.choices[0].message.content
    lines = response.split('\n')
    task = lines[0]
    
    print(f"TASK: {task}")
    retval = {}
    if task == "Heatmap":
        par_list = eval(lines[1])
        column = par_list[0]
        if column == 'bouguer_an' or column== 'elevation_':
            retval= create_heatmap(gravity_file, column, lat_min, lat_max, long_min, long_max)
        elif column=='magnetic_a':
            retval= create_heatmap(magnetic_file, column, lat_min, lat_max, long_min, long_max)
        else:
            retval= create_heatmap(stream_sed_file, column, lat_min, lat_max, long_min, long_max)
    
    elif task== "Contour":
        par_list = eval(lines[1])
        column = par_list[0]
        if column == 'bouguer_an' or column== 'elevation_':
            retval= create_contour_map(gravity_file, column, lat_min, lat_max, long_min, long_max)
        elif column=='magnetic_a':
            retval= create_contour_map(magnetic_file, column, lat_min, lat_max, long_min, long_max)
        else:
            retval= create_contour_map(stream_sed_file, column, lat_min, lat_max, long_min, long_max)
    
    elif task=="IDWInterpolation":
        par_list = eval(lines[1])
        column = par_list[0]
        if column == 'bouguer_an' or column== 'elevation_':
            retval= interpolation_geojson(gravity_file, column, lat_min, lat_max, long_min, long_max, 1)
        elif column=='magnetic_a':
            retval= interpolation_geojson(magnetic_file, column, lat_min, lat_max, long_min, long_max, 1)
        else:
            retval= interpolation_geojson(stream_sed_file, column, lat_min, lat_max, long_min, long_max, 1)
    
    elif task=="SplineInterpolation":
        par_list = eval(lines[1])
        column = par_list[0]
        if column == 'bouguer_an' or column== 'elevation_':
            retval= interpolation_geojson(gravity_file, column, lat_min, lat_max, long_min, long_max, 2)
        elif column=='magnetic_a':
            retval= interpolation_geojson(magnetic_file, column, lat_min, lat_max, long_min, long_max, 2)
        else:
            retval= interpolation_geojson(stream_sed_file, column, lat_min, lat_max, long_min, long_max, 2)
    
    elif task=="KrigingInterpolation":
        par_list = eval(lines[1])
        column = par_list[0]
        if column == 'bouguer_an' or column== 'elevation_':
            retval= interpolation_geojson(gravity_file, column, lat_min, lat_max, long_min, long_max, 3)
        elif column=='magnetic_a':
            retval= interpolation_geojson(magnetic_file, column, lat_min, lat_max, long_min, long_max, 3)
        else:
            retval= interpolation_geojson(stream_sed_file, column, lat_min, lat_max, long_min, long_max, 3)

    elif task=="NNInterpolation":
        par_list = eval(lines[1])
        column = par_list[0]
        if column == 'bouguer_an' or column== 'elevation_':
            retval= interpolation_geojson(gravity_file, column, lat_min, lat_max, long_min, long_max, 4)
        elif column=='magnetic_a':
            retval= interpolation_geojson(magnetic_file, column, lat_min, lat_max, long_min, long_max, 4)
        else:
            retval= interpolation_geojson(stream_sed_file, column, lat_min, lat_max, long_min, long_max, 4)
    
    
    elif task == "Stat":
        par_list = eval(lines[1])
        column = par_list[0]
        retval= plot_high_value_points(stream_sed_file, column, lat_min, lat_max, long_min, long_max)
    
    elif task == "Exploration":
        retval=plot_excavation_sites(exploration_file, lat_min, lat_max, long_min, long_max)

    elif task == "Histogram":
        par_list = eval(lines[1])
        retval= create_geochem_avg_histogram(stream_sed_file,par_list, lat_min, lat_max, long_min, long_max)

    else:
        retval = {"message": response, "html": None}


    return retval








stream_sed_file = r'ngdr_json\stream_sediments_chem.geojson'
gravity_file = r'ngdr_json\gravity_phy.geojson'
magnetic_file = r'ngdr_json\magnetic_phy.geojson'
exploration_file = r'ngdr_json\exploration_data.geojson'
# iface= gr.Interface(fn = Manager_agent,
#                     inputs = ["text","text","text","text","text"],
#                     outputs = "json",
#                     title = "DataExplorer_Agent",
#                     description="Gets Geological textual and visual outputs for a query")

# iface.launch(inline=False)

latmin = 21.0
latmax = 22.0
longmin = 78.0
longmax = 79.0
mquery = ""


response = Manager_agent(mquery, latmin, latmax, longmin, longmax)

if "html" in response:
    with open('heatmap.html', 'w') as f:
        f.write(response["html"])
    print(response["message"])
else:
    print(response["message"])




# To automatically open the HTML file in the default web browser (optional)
import webbrowser
import os

webbrowser.open(f"heatmap.html")