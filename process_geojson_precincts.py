import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.wkt import loads
import folium
import logging
import os
import requests
import matplotlib.pyplot as plt
from branca.colormap import linear
import json  # Added for GeoJSON validation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_precincts_data(geojson_url="https://raw.githubusercontent.com/jkeefe/census-by-precincts/master/data/nyc/nyc_precincts.geojson", local_path="data/nyc_precincts.geojson"):
    """
    Load the NYC Police Precincts GeoJSON data and validate its format.
    
    Parameters:
    -----------
    geojson_url : str
        URL to the GeoJSON file.
    local_path : str
        Local path to save the downloaded GeoJSON file.
        
    Returns:
    --------
    geopandas.GeoDataFrame
        A GeoDataFrame containing precinct boundaries and identifiers.
    """
    try:
        # Download GeoJSON if not present locally
        if not os.path.exists(local_path):
            logging.info(f"Downloading GeoJSON from {geojson_url}")
            response = requests.get(geojson_url)
            response.raise_for_status()
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, 'wb') as f:
                f.write(response.content)
            logging.info(f"Saved GeoJSON to {local_path}")
        
        # Validate GeoJSON format
        logging.info(f"Validating GeoJSON file: {local_path}")
        with open(local_path, 'r') as f:
            try:
                json.load(f)  # Attempt to parse the file as JSON
                logging.info("GeoJSON file is valid!")
            except json.JSONDecodeError as e:
                logging.error(f"Invalid GeoJSON format: {str(e)}")
                raise ValueError(f"GeoJSON file at {local_path} is malformed: {str(e)}")
        
        # Load GeoJSON into GeoDataFrame
        precincts_gdf = gpd.read_file(local_path)
        if precincts_gdf.empty:
            raise ValueError("Precincts dataset is empty")
        
        logging.info(f"Loaded {len(precincts_gdf)} precincts with columns: {', '.join(precincts_gdf.columns)}")
        
        # Ensure the precinct identifier column exists (assuming 'precinct')
        if 'precinct' not in precincts_gdf.columns:
            raise ValueError("GeoJSON missing 'precinct' column")
        
        return precincts_gdf
    
    except Exception as e:
        logging.error(f"Error loading precincts data: {str(e)}")
        raise

# [Rest of the functions remain unchanged for brevity; they are assumed to work as provided]
def merge_calls_and_precincts(calls_df, precincts_df, calls_col="PRECINCT", precincts_col="precinct"):
    logging.info("Merging calls and precincts data")
    calls_by_precinct = calls_df.groupby(calls_col)['call_count'].sum().reset_index()
    logging.info(f"Aggregated calls for {len(calls_by_precinct)} precincts")
    merged_df = precincts_df.merge(calls_by_precinct, left_on=precincts_col, right_on=calls_col, how='left')
    if 'call_count' in merged_df.columns:
        null_counts = merged_df['call_count'].isnull().sum()
        if null_counts > 0:
            logging.warning(f"Found {null_counts} precincts without call data. Filling with 0.")
        merged_df['call_count'] = merged_df['call_count'].fillna(0)
    else:
        logging.warning("No 'call_count' column in merged data. Adding with default value 0.")
        merged_df['call_count'] = 0
    logging.info(f"Successfully matched {len(merged_df[merged_df['call_count'] > 0])} precincts with call data out of {len(merged_df)} total precincts")
    return merged_df

def process_geospatial_data(merged_df):
    logging.info("Processing geospatial data")
    if not isinstance(merged_df, gpd.GeoDataFrame) or 'geometry' not in merged_df.columns:
        raise ValueError("Merged dataset is not a valid GeoDataFrame with geometries")
    if merged_df.geometry.isnull().any():
        null_geoms = merged_df.geometry.isnull().sum()
        logging.warning(f"Found {null_geoms} null geometries in GeoDataFrame. Removing them.")
        merged_df = merged_df[merged_df.geometry.notna()]
    logging.info("Geospatial data processed successfully")
    return merged_df

def visualize_call_volumes(merged_df, output_path="eda_plots/call_volume_map.html"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not isinstance(merged_df, gpd.GeoDataFrame) or 'geometry' not in merged_df.columns:
        logging.warning("Cannot create map visualization: Input is not a valid GeoDataFrame with geometries")
        return
    logging.info("Creating choropleth map for call volumes by precinct")
    merged_df = merged_df.to_crs(epsg=4326)
    nyc_map = folium.Map(location=[40.7128, -74.0060], zoom_start=10, tiles='cartodbpositron')
    folium.Choropleth(
        geo_data=merged_df.__geo_interface__,
        name='Call Volumes',
        data=merged_df,
        columns=['precinct', 'call_count'],
        key_on='feature.properties.precinct',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Number of Calls'
    ).add_to(nyc_map)
    folium.features.GeoJson(
        merged_df,
        name='Precincts',
        tooltip=folium.features.GeoJsonTooltip(
            fields=['precinct', 'call_count'],
            aliases=['Precinct:', 'Number of Calls:'],
            localize=True
        )
    ).add_to(nyc_map)
    folium.LayerControl().add_to(nyc_map)
    nyc_map.save(output_path)
    logging.info(f"Choropleth map saved to {output_path}")

def standardize_precincts(precincts_df, calls_df, precincts_col="precinct", calls_col="PRECINCT"):
    logging.info("Standardizing precinct identifiers")
    precincts_df[precincts_col] = precincts_df[precincts_col].astype(str)
    calls_df[calls_col] = calls_df[calls_col].astype(str).str.replace(r'\.0$', '', regex=True)
    calls_precincts = set(calls_df[calls_col].unique())
    precincts = set(precincts_df[precincts_col].unique())
    missing_in_precincts = calls_precincts - precincts
    if missing_in_precincts:
        logging.warning(f"Found {len(missing_in_precincts)} precincts in calls data not in precincts data: {missing_in_precincts}")
    extra_in_precincts = precincts - calls_precincts
    if extra_in_precincts:
        logging.warning(f"Found {len(extra_in_precincts)} precincts in precincts data not in calls data: {extra_in_precincts}")
    logging.info("Precinct identifiers standardized")
    return precincts_df, calls_df

def preprocess_and_integrate_geospatial(calls_path="data/grouped_data.csv"):
    logging.info("Starting geospatial preprocessing and integration")
    precincts_df = load_precincts_data()
    try:
        logging.info(f"Loading calls data from {calls_path}")
        calls_df = pd.read_csv(calls_path)
        logging.info(f"Loaded calls data with {len(calls_df)} records and columns: {', '.join(calls_df.columns)}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Calls data file not found: {calls_path}")
    precincts_df, calls_df = standardize_precincts(precincts_df, calls_df)
    merged_df = merge_calls_and_precincts(calls_df, precincts_df)
    geo_df = process_geospatial_data(merged_df)
    visualize_call_volumes(geo_df)
    output_path = "data/merged_calls_precincts.csv"
    geo_df.to_csv(output_path, index=False)
    logging.info(f"Merged data saved to {output_path}")
    geojson_path = "data/merged_calls_precincts.geojson"
    geo_df.to_file(geojson_path, driver="GeoJSON")
    logging.info(f"GeoJSON data saved to {geojson_path}")
    logging.info("Geospatial preprocessing and integration complete")
    return geo_df

if __name__ == "__main__":
    try:
        merged_gdf = preprocess_and_integrate_geospatial()
        print("\nFirst 5 rows of merged data:")
        print(merged_gdf.head())
        print("\nSummary statistics for call volumes by precinct:")
        print(merged_gdf['call_count'].describe())
        print(f"\nTotal number of precincts: {len(merged_gdf)}")
        print(f"Total number of calls: {merged_gdf['call_count'].sum()}")
        print(f"Precincts with highest call volumes:")
        top_precincts = merged_gdf.sort_values('call_count', ascending=False).head(5)
        for _, row in top_precincts.iterrows():
            print(f"  Precinct {row['precinct']}: {int(row['call_count']):,} calls")
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise