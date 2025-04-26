import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.wkt import loads
import folium
import logging
import os
import matplotlib.pyplot as plt
from branca.colormap import linear

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_precincts_data(file_path="data/NYC_Police_Precincts_2585695823843699054.csv"):
    """
    Load the NYC Police Precincts dataset.
    
    Parameters:
    -----------
    file_path : str
        Path to the precincts dataset CSV file.
        
    Returns:
    --------
    geopandas.GeoDataFrame or pandas.DataFrame
        A GeoDataFrame if geospatial data is available, otherwise a DataFrame.
        
    Example:
    --------
    >>> precincts_gdf = load_precincts_data()
    >>> print(precincts_gdf.head())
    """
    try:
        logging.info(f"Loading precincts data from {file_path}")
        precincts_df = pd.read_csv(file_path)
        
        if precincts_df.empty:
            raise ValueError("Precincts dataset is empty")
        
        logging.info(f"Loaded {len(precincts_df)} precincts with columns: {', '.join(precincts_df.columns)}")
        
        # Check if the dataset has a geometry column or can be converted to a GeoDataFrame
        if 'geometry' in precincts_df.columns:
            precincts_gdf = gpd.GeoDataFrame(
                precincts_df, 
                geometry=precincts_df['geometry'].apply(loads), 
                crs="EPSG:4326"
            )
            logging.info(f"Converted to GeoDataFrame with {len(precincts_gdf)} geometries")
            return precincts_gdf
        else:
            logging.warning("No explicit geometry column found in precincts data")
            # For this specific dataset, we know it has Shape__Area and Shape__Length but no geometry
            # We'll handle this in the process_geospatial_data function if needed
            return precincts_df
    
    except FileNotFoundError:
        error_msg = f"Precincts file not found: {file_path}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)
    except Exception as e:
        error_msg = f"Error loading precincts data: {str(e)}"
        logging.error(error_msg)
        raise

def standardize_precincts(precincts_df, calls_df, precincts_col="Precinct", calls_col="PRECINCT"):
    """
    Standardize precinct identifiers to ensure consistency between datasets.
    
    Parameters:
    -----------
    precincts_df : pandas.DataFrame or geopandas.GeoDataFrame
        The precincts dataset.
    calls_df : pandas.DataFrame
        The calls for service dataset.
    precincts_col : str
        The column name containing precinct identifiers in precincts_df.
    calls_col : str
        The column name containing precinct identifiers in calls_df.
        
    Returns:
    --------
    tuple
        (precincts_df, calls_df) with standardized precinct identifiers.
        
    Example:
    --------
    >>> precincts_df, calls_df = standardize_precincts(precincts_df, calls_df)
    """
    logging.info("Standardizing precinct identifiers")
    
    # Ensure precinct columns exist
    if precincts_col not in precincts_df.columns:
        error_msg = f"Column '{precincts_col}' not found in precincts dataset"
        logging.error(error_msg)
        raise ValueError(error_msg)
        
    if calls_col not in calls_df.columns:
        error_msg = f"Column '{calls_col}' not found in calls dataset"
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    # Convert to string type for consistency
    precincts_df[precincts_col] = precincts_df[precincts_col].astype(str)
    calls_df[calls_col] = calls_df[calls_col].astype(str)
    
    # Check for precincts in calls data not in precincts data
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

def merge_calls_and_precincts(calls_df, precincts_df, calls_col="PRECINCT", precincts_col="Precinct"):
    """
    Aggregate call counts by precinct and merge with precinct data.
    
    Parameters:
    -----------
    calls_df : pandas.DataFrame
        The calls for service dataset with columns like date, hour, PRECINCT, TYP_DESC, call_count.
    precincts_df : pandas.DataFrame or geopandas.GeoDataFrame
        The precincts dataset.
    calls_col : str
        The column name containing precinct identifiers in calls_df.
    precincts_col : str
        The column name containing precinct identifiers in precincts_df.
        
    Returns:
    --------
    pandas.DataFrame or geopandas.GeoDataFrame
        A merged dataset containing call counts per precinct.
        
    Example:
    --------
    >>> merged_df = merge_calls_and_precincts(calls_df, precincts_gdf)
    """
    logging.info("Merging calls and precincts data")
    
    # Aggregate calls by precinct
    calls_by_precinct = calls_df.groupby(calls_col)['call_count'].sum().reset_index()
    logging.info(f"Aggregated calls for {len(calls_by_precinct)} precincts")
    
    # Perform left join from precincts to calls
    merged_df = precincts_df.merge(
        calls_by_precinct,
        left_on=precincts_col,
        right_on=calls_col,
        how='left'
    )
    
    # Fill NaN values in call_count with 0
    if 'call_count' in merged_df.columns:
        null_counts = merged_df['call_count'].isnull().sum()
        if null_counts > 0:
            logging.warning(f"Found {null_counts} precincts without call data. Filling with 0.")
        merged_df['call_count'] = merged_df['call_count'].fillna(0)
    else:
        logging.warning("No 'call_count' column in merged data. Adding with default value 0.")
        merged_df['call_count'] = 0
    
    # Log merge results
    matched_precincts = len(merged_df[merged_df['call_count'] > 0])
    logging.info(f"Successfully matched {matched_precincts} precincts with call data out of {len(merged_df)} total precincts")
    
    return merged_df

def process_geospatial_data(merged_df):
    """
    Ensure the merged dataset has properly formatted geospatial data.
    
    Parameters:
    -----------
    merged_df : pandas.DataFrame or geopandas.GeoDataFrame
        The merged dataset to process.
        
    Returns:
    --------
    geopandas.GeoDataFrame or pandas.DataFrame
        A dataset with geospatial data if available.
        
    Example:
    --------
    >>> geo_df = process_geospatial_data(merged_df)
    """
    logging.info("Processing geospatial data")
    
    # Check if already a GeoDataFrame with geometry column
    if isinstance(merged_df, gpd.GeoDataFrame) and merged_df.geometry.notna().all():
        logging.info("Dataset is already a valid GeoDataFrame")
        return merged_df
    
    # Check if it's a GeoDataFrame but with some null geometries
    if isinstance(merged_df, gpd.GeoDataFrame) and merged_df.geometry.isnull().any():
        null_geoms = merged_df.geometry.isnull().sum()
        logging.warning(f"Found {null_geoms} null geometries in GeoDataFrame")
        # Remove rows with null geometries to avoid issues with visualization
        merged_df = merged_df[merged_df.geometry.notna()]
        return merged_df

    # Check for latitude/longitude columns
    lat_cols = [col for col in merged_df.columns if 'lat' in col.lower()]
    lon_cols = [col for col in merged_df.columns if 'lon' in col.lower()]
    
    if lat_cols and lon_cols:
        logging.info(f"Found latitude column '{lat_cols[0]}' and longitude column '{lon_cols[0]}'")
        try:
            points = [Point(xy) for xy in zip(merged_df[lon_cols[0]], merged_df[lat_cols[0]])]
            merged_gdf = gpd.GeoDataFrame(merged_df, geometry=points, crs="EPSG:4326")
            logging.info(f"Created geometry from lat/lon coordinates for {len(merged_gdf)} records")
            return merged_gdf
        except Exception as e:
            logging.error(f"Error creating geometry from lat/lon: {str(e)}")
            
    # For this specific dataset, we know it has Shape__Area and Shape__Length but may not have explicit geometry
    # In a real scenario, you would need more info about the dataset format or external boundary files
    logging.warning("No usable geospatial data found in the merged dataset")
    logging.warning("For a complete implementation, you would need to obtain precinct boundary files")
    
    return merged_df

def visualize_call_volumes(merged_df, output_path="eda_plots/call_volume_map.html"):
    """
    Create a choropleth map to visualize call volumes by precinct.
    
    Parameters:
    -----------
    merged_df : pandas.DataFrame or geopandas.GeoDataFrame
        The merged dataset with call counts and precinct data.
    output_path : str
        Path where the visualization will be saved.
        
    Returns:
    --------
    None
        Saves the visualization to the specified file path.
        
    Example:
    --------
    >>> visualize_call_volumes(merged_gdf)
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if not isinstance(merged_df, gpd.GeoDataFrame) or 'geometry' not in merged_df.columns:
        logging.warning("Cannot create map visualization: Input is not a valid GeoDataFrame with geometries")
        
        # Create a simple bar chart as an alternative visualization
        plt.figure(figsize=(12, 8))
        
        # Use the actual precinct column name from your dataset
        precinct_col = "Precinct" if "Precinct" in merged_df.columns else merged_df.columns[0]
        
        # Sort by call volume and get top 20
        plot_df = merged_df.sort_values('call_count', ascending=False).head(20)
        
        plt.bar(plot_df[precinct_col], plot_df['call_count'], color='navy')
        plt.title('Top 20 Precincts by Call Volume')
        plt.xlabel('Precinct')
        plt.ylabel('Number of Calls')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save as PNG instead of interactive map
        alt_output = output_path.replace('.html', '.png')
        plt.savefig(alt_output)
        logging.info(f"Created alternative bar chart visualization at {alt_output}")
        return
    
    logging.info("Creating choropleth map for call volumes by precinct")
    
    # Convert the GeoDataFrame to GeoJSON for folium
    merged_df = merged_df.to_crs(epsg=4326)  # Ensure WGS84 projection for web mapping
    
    # Create base map centered on NYC
    nyc_map = folium.Map(
        location=[40.7128, -74.0060],
        zoom_start=10,
        tiles='cartodbpositron'
    )
    
    # Get min and max values for colormap
    min_calls = merged_df['call_count'].min()
    max_calls = merged_df['call_count'].max()
    
    # Create a color map
    colormap = linear.YlOrRd.scale(min_calls, max_calls)
    
    # Add the colormap to the map
    colormap.caption = 'Call Volume by Precinct'
    nyc_map.add_child(colormap)
    
    # Create the choropleth layer
    folium.Choropleth(
        geo_data=merged_df.__geo_interface__,
        name='Call Volumes',
        data=merged_df,
        columns=['Precinct', 'call_count'],
        key_on='feature.properties.Precinct',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Number of Calls'
    ).add_to(nyc_map)
    
    # Add tooltips
    folium.features.GeoJson(
        merged_df,
        name='Precincts',
        tooltip=folium.features.GeoJsonTooltip(
            fields=['Precinct', 'call_count'],
            aliases=['Precinct:', 'Number of Calls:'],
            localize=True
        )
    ).add_to(nyc_map)
    
    # Add layer control
    folium.LayerControl().add_to(nyc_map)
    
    # Save the map
    nyc_map.save(output_path)
    logging.info(f"Choropleth map saved to {output_path}")

def preprocess_and_integrate_geospatial(precincts_path="data/NYC_Police_Precincts_2585695823843699054.csv", calls_path="data/grouped_data.csv"):
    """
    Orchestrate preprocessing and integration of geospatial data.
    
    Parameters:
    -----------
    precincts_path : str
        Path to the precincts dataset CSV file.
    calls_path : str
        Path to the preprocessed calls dataset CSV file.
        
    Returns:
    --------
    pandas.DataFrame or geopandas.GeoDataFrame
        A merged dataset with call counts and precinct information.
        
    Example:
    --------
    >>> merged_data = preprocess_and_integrate_geospatial()
    >>> print(merged_data.head())
    """
    logging.info("Starting geospatial preprocessing and integration")
    
    # Load precincts data
    precincts_df = load_precincts_data(precincts_path)
    
    # Load calls data
    try:
        logging.info(f"Loading calls data from {calls_path}")
        calls_df = pd.read_csv(calls_path)
        logging.info(f"Loaded calls data with {len(calls_df)} records and columns: {', '.join(calls_df.columns)}")
    except FileNotFoundError:
        error_msg = f"Calls data file not found: {calls_path}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Standardize precinct identifiers
    precincts_df, calls_df = standardize_precincts(precincts_df, calls_df)
    
    # Merge datasets
    merged_df = merge_calls_and_precincts(calls_df, precincts_df)
    
    # Process geospatial data
    geo_df = process_geospatial_data(merged_df)
    
    # Create visualization
    visualize_call_volumes(geo_df)
    
    # Save merged data
    output_path = "data/merged_calls_precincts.csv"
    geo_df.to_csv(output_path, index=False)
    logging.info(f"Merged data saved to {output_path}")
    
    if isinstance(geo_df, gpd.GeoDataFrame):
        # Also save as GeoJSON for potential use in web applications
        geojson_path = "data/merged_calls_precincts.geojson"
        try:
            geo_df.to_file(geojson_path, driver="GeoJSON")
            logging.info(f"GeoJSON data saved to {geojson_path}")
        except Exception as e:
            logging.warning(f"Could not save GeoJSON: {str(e)}")
    
    logging.info("Geospatial preprocessing and integration complete")
    return geo_df

if __name__ == "__main__":
    try:
        merged_gdf = preprocess_and_integrate_geospatial()
        print("\nFirst 5 rows of merged data:")
        print(merged_gdf.head())
        
        print("\nSummary statistics for call volumes by precinct:")
        print(merged_gdf['call_count'].describe())
        
        # Additional information about the dataset
        print(f"\nTotal number of precincts: {len(merged_gdf)}")
        print(f"Total number of calls: {merged_gdf['call_count'].sum()}")
        print(f"Precincts with highest call volumes:")
        top_precincts = merged_gdf.sort_values('call_count', ascending=False).head(5)
        for _, row in top_precincts.iterrows():
            print(f"  Precinct {row['Precinct']}: {int(row['call_count']):,} calls")
            
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise
