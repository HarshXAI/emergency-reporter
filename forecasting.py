import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import folium
from folium.plugins import HeatMap
from prophet import Prophet
from datetime import datetime, timedelta
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger("NYPD-Forecasting")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# Create output directories if they don't exist
Path("data").mkdir(exist_ok=True)
Path("eda_plots").mkdir(exist_ok=True)

def load_data():
    """
    Load the grouped call data and merged calls-precincts data.
    """
    try:
        logger.info("Loading datasets...")
        # Load time-series data
        grouped_df = pd.read_csv('data/grouped_data.csv')
        # Convert date to datetime
        grouped_df['date'] = pd.to_datetime(grouped_df['date'])
        
        # Standardize PRECINCT column early to ensure consistency
        if 'PRECINCT' in grouped_df.columns:
            grouped_df['PRECINCT'] = grouped_df['PRECINCT'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            logger.info(f"Unique PRECINCT values in grouped_df: {sorted(grouped_df['PRECINCT'].unique().tolist())}")
        
        # Load geospatial data
        precinct_df = gpd.read_file('/Users/harshkanani/Desktop/Spit/data/merged_calls_precincts.geojson')
        
        # Standardize precinct column in geospatial data
        if 'precinct' in precinct_df.columns:
            precinct_df['precinct'] = precinct_df['precinct'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            logger.info(f"Unique precinct values in precinct_df: {sorted(precinct_df['precinct'].unique().tolist())}")
        else:
            # Check for alternative capitalization
            potential_columns = [col for col in precinct_df.columns if col.lower() == 'precinct']
            if potential_columns:
                logger.info(f"Found alternative precinct column: {potential_columns[0]}")
                precinct_df = precinct_df.rename(columns={potential_columns[0]: 'precinct'})
                precinct_df['precinct'] = precinct_df['precinct'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            else:
                logger.error("No precinct column found in geospatial data")
        
        logger.info("Data loaded successfully")
        return grouped_df, precinct_df
    except FileNotFoundError as e:
        logger.error(f"Failed to load data files: {e}")
        raise

def explore_historical_data(grouped_df):
    """
    Analyze historical emergency call patterns and create visualizations.
    """
    logger.info("Analyzing historical call patterns...")
    
    # Check dataframe structure before aggregating
    logger.info(f"Columns in dataframe: {grouped_df.columns.tolist()}")
    logger.info(f"Data types: {grouped_df.dtypes}")
    
    # Ensure date is datetime and hour is numeric
    grouped_df['date'] = pd.to_datetime(grouped_df['date'])
    grouped_df['hour'] = pd.to_numeric(grouped_df['hour'], errors='coerce')
    
    # Aggregate call counts by precinct, date, and hour - only sum the call_count column
    hourly_agg = grouped_df.groupby(['date', 'hour']).agg({'call_count': 'sum'}).reset_index()
    daily_agg = grouped_df.groupby('date').agg({'call_count': 'sum'}).reset_index()
    
    # Plot hourly trends
    plt.figure(figsize=(12, 6))
    avg_hourly = hourly_agg.groupby('hour').mean()['call_count']
    sns.lineplot(x=avg_hourly.index, y=avg_hourly.values)
    plt.title('Average Call Volume by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Call Count')
    plt.xticks(range(0, 24))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('eda_plots/hourly_trends.png')
    plt.close()
    
    # Plot daily trends
    plt.figure(figsize=(14, 6))
    sns.lineplot(x=daily_agg['date'], y=daily_agg['call_count'])
    plt.title('Daily Call Volume Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Call Count')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('eda_plots/daily_trends.png')
    plt.close()
    
    # Summarize top call types
    plt.figure(figsize=(12, 8))
    call_types = grouped_df.groupby('TYP_DESC').agg({'call_count': 'sum'}).sort_values('call_count', ascending=False).head(5)
    sns.barplot(x=call_types['call_count'], y=call_types.index)
    plt.title('Top 5 Emergency Call Types')
    plt.xlabel('Total Call Count')
    plt.ylabel('Call Type Description')
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('eda_plots/call_types.png')
    plt.close()
    
    logger.info("Historical data analysis completed")
    return hourly_agg, daily_agg

def detect_anomalies(grouped_df):
    """
    Detect anomalies in daily call volumes using z-scores.
    """
    logger.info("Detecting anomalies in call volumes...")
    
    # Aggregate by date and precinct
    daily_precinct = grouped_df.groupby(['date', 'PRECINCT']).sum()['call_count'].reset_index()
    
    # Initialize anomaly dataframe
    anomalies = []
    
    # Detect anomalies for each precinct
    for precinct in daily_precinct['PRECINCT'].unique():
        precinct_data = daily_precinct[daily_precinct['PRECINCT'] == precinct]
        
        # Calculate z-scores
        mean_calls = precinct_data['call_count'].mean()
        std_calls = precinct_data['call_count'].std()
        
        if std_calls == 0:
            # Skip precincts with constant call volume
            continue
            
        precinct_data['z_score'] = (precinct_data['call_count'] - mean_calls) / std_calls
        
        # Identify anomalies (z-score > 2 or < -2)
        precinct_anomalies = precinct_data[abs(precinct_data['z_score']) > 2]
        
        if not precinct_anomalies.empty:
            anomalies.append(precinct_anomalies)
    
    if anomalies:
        all_anomalies = pd.concat(anomalies)
        all_anomalies.to_csv('data/call_anomalies.csv', index=False)
        logger.info(f"Found {len(all_anomalies)} anomalous days across precincts")
    else:
        pd.DataFrame(columns=['date', 'PRECINCT', 'call_count', 'z_score']).to_csv('data/call_anomalies.csv', index=False)
        logger.info("No anomalies detected")
    
    return anomalies

def forecast_call_volumes(grouped_df):
    """
    Forecast call volumes for each precinct using Facebook Prophet.
    """
    logger.info("Forecasting call volumes...")
    
    # Aggregate data by date and precinct
    daily_precinct = grouped_df.groupby(['date', 'PRECINCT']).sum()['call_count'].reset_index()
    daily_precinct['ds'] = daily_precinct['date']  # Prophet requires 'ds' column
    daily_precinct['y'] = daily_precinct['call_count']  # Prophet requires 'y' column
    
    all_forecasts = []
    precincts = daily_precinct['PRECINCT'].unique()
    
    for precinct in precincts:
        try:
            logger.info(f"Forecasting for Precinct {precinct}...")
            precinct_data = daily_precinct[daily_precinct['PRECINCT'] == precinct]
            
            # Create and fit Prophet model
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
                interval_width=0.95,
                seasonality_mode='multiplicative'
            )
            model.fit(precinct_data[['ds', 'y']])
            
            # Create future dataframe for 7 days
            future = model.make_future_dataframe(periods=7, freq='D')
            forecast = model.predict(future)
            
            # Add precinct information to forecast - ensure it's a standardized string format
            forecast['PRECINCT'] = str(precinct).replace('.0', '').strip()
            
            # Keep forecasts for future days only
            max_date = precinct_data['ds'].max()
            forecast = forecast[forecast['ds'] > max_date]
            
            # Select relevant columns
            forecast_subset = forecast[['ds', 'PRECINCT', 'yhat', 'yhat_lower', 'yhat_upper']]
            forecast_subset.columns = ['date', 'PRECINCT', 'predicted_calls', 'lower_bound', 'upper_bound']
            all_forecasts.append(forecast_subset)
            
        except Exception as e:
            logger.warning(f"Forecasting failed for Precinct {precinct}: {e}")
            continue
    
    if all_forecasts:
        forecasted_calls = pd.concat(all_forecasts)
        forecasted_calls['predicted_calls'] = forecasted_calls['predicted_calls'].apply(lambda x: max(0, round(x)))
        forecasted_calls['lower_bound'] = forecasted_calls['lower_bound'].apply(lambda x: max(0, round(x)))
        forecasted_calls['upper_bound'] = forecasted_calls['upper_bound'].apply(lambda x: max(0, round(x)))
        
        forecasted_calls.to_csv('data/forecasted_calls.csv', index=False)
        logger.info(f"Generated forecasts for {len(precincts)} precincts")
        
        return forecasted_calls
    else:
        logger.error("Failed to generate forecasts for any precinct")
        return None

def create_geospatial_prediction(forecasted_calls, precinct_df):
    """
    Create a choropleth map visualizing predicted call volumes by precinct.
    
    Parameters:
    - forecasted_calls: DataFrame with columns ['PRECINCT', 'predicted_calls', 'lower_bound', 'upper_bound']
    - precinct_df: GeoDataFrame with columns ['precinct', 'geometry', 'shape_area', 'shape_leng']
    
    Returns:
    - geo_forecast: Merged GeoDataFrame with geospatial and forecast data
    """
    logger.info("Creating geospatial prediction map...")
    
    try:
        # Log input data for debugging
        logger.info(f"Forecasted calls columns: {forecasted_calls.columns.tolist()}")
        logger.info(f"Forecasted calls data types: {forecasted_calls.dtypes}")
        logger.info(f"Precinct df columns: {precinct_df.columns.tolist()}")
        logger.info(f"Precinct df data types: {precinct_df['precinct'].dtype}")
        
        # Standardize PRECINCT column in forecasted_calls
        forecasted_calls['PRECINCT'] = forecasted_calls['PRECINCT'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        
        # Standardize precinct column in precinct_df
        precinct_df['precinct'] = precinct_df['precinct'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        
        # Log sample values to debug potential mismatches
        logger.info(f"Sample PRECINCT values in forecasted_calls: {forecasted_calls['PRECINCT'].head().tolist()}")
        logger.info(f"Sample precinct values in precinct_df: {precinct_df['precinct'].head().tolist()}")
        
        # Aggregate forecasted calls by precinct
        precinct_forecast = forecasted_calls.groupby('PRECINCT').agg({
            'predicted_calls': 'sum',
            'lower_bound': 'sum',
            'upper_bound': 'sum'
        }).reset_index()
        
        # Rename and standardize
        precinct_forecast.rename(columns={'PRECINCT': 'precinct'}, inplace=True)
        precinct_forecast['precinct'] = precinct_forecast['precinct'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        
        # Check for any missing or extra precincts between datasets
        forecast_precincts = set(precinct_forecast['precinct'].unique())
        geo_precincts = set(precinct_df['precinct'].unique())
        
        missing_in_geo = forecast_precincts - geo_precincts
        missing_in_forecast = geo_precincts - forecast_precincts
        
        if missing_in_geo:
            logger.warning(f"Precincts in forecast but missing in geospatial data: {missing_in_geo}")
        
        if missing_in_forecast:
            logger.warning(f"Precincts in geospatial data but missing in forecast: {missing_in_forecast}")
        
        # Merge with geospatial data - use outer join to diagnose merging issues
        pre_merge_forecast_sum = precinct_forecast['predicted_calls'].sum()
        logger.info(f"Sum of predicted calls before merge: {pre_merge_forecast_sum}")
        
        # First do an inner join to check matches
        inner_merge = precinct_df.merge(precinct_forecast, on='precinct', how='inner')
        logger.info(f"Number of precincts matched in inner merge: {len(inner_merge)}")
        logger.info(f"Sum of predicted calls after inner merge: {inner_merge['predicted_calls'].sum()}")
        
        # If inner merge is empty or significant data loss, log details
        if len(inner_merge) == 0 or inner_merge['predicted_calls'].sum() < pre_merge_forecast_sum * 0.9:
            logger.error("Significant data loss in merge. Detailed match evaluation:")
            # Compare a few example values
            for p_idx, p_val in enumerate(precinct_forecast['precinct'].head()):
                found = p_val in precinct_df['precinct'].values
                logger.info(f"Precinct {p_val} from forecast {'found' if found else 'NOT FOUND'} in geo data")
                logger.info(f"   Value type: {type(p_val)}, Length: {len(p_val)}")
                if not found:
                    # Try to find close matches for debugging
                    for geo_val in precinct_df['precinct'].unique():
                        if p_val in geo_val or geo_val in p_val:
                            logger.info(f"   Potential match: '{geo_val}' (type: {type(geo_val)}, length: {len(geo_val)})")
        
        # Now do the actual left merge for the final result
        geo_forecast = precinct_df.merge(precinct_forecast, on='precinct', how='left')
        
        # Ensure geo_forecast is a GeoDataFrame
        if not isinstance(geo_forecast, gpd.GeoDataFrame):
            geo_forecast = gpd.GeoDataFrame(geo_forecast, geometry='geometry')
        
        # Fill NaN values for forecast columns
        geo_forecast['predicted_calls'] = geo_forecast['predicted_calls'].fillna(0)
        geo_forecast['lower_bound'] = geo_forecast['lower_bound'].fillna(0)
        geo_forecast['upper_bound'] = geo_forecast['upper_bound'].fillna(0)
        
        # Log success rate of the merge
        total_predicted = geo_forecast['predicted_calls'].sum()
        logger.info(f"Total predicted calls after merge: {total_predicted}")
        
        if total_predicted == 0:
            logger.error("No predicted calls available after merge. Merge likely failed completely.")
            
            # Try a drastic approach - direct assignment if possible
            if len(precinct_forecast) == len(precinct_df):
                logger.info("Attempting direct assignment as last resort...")
                # Sort both dataframes by precinct
                precinct_forecast = precinct_forecast.sort_values('precinct').reset_index(drop=True)
                geo_forecast = geo_forecast.sort_values('precinct').reset_index(drop=True)
                
                # Directly assign values
                geo_forecast['predicted_calls'] = precinct_forecast['predicted_calls'].values
                geo_forecast['lower_bound'] = precinct_forecast['lower_bound'].values
                geo_forecast['upper_bound'] = precinct_forecast['upper_bound'].values
                
                logger.info(f"After direct assignment: {geo_forecast['predicted_calls'].sum()} total predicted calls")
        
        # Validate geometry column
        if 'geometry' not in geo_forecast.columns:
            logger.error("'geometry' column is missing in geo_forecast")
            return None
        
        # Check for missing or invalid geometries
        if geo_forecast['geometry'].isnull().any():
            logger.error(f"Missing geometries found: {geo_forecast[geo_forecast['geometry'].isnull()]['precinct'].tolist()}")
            return None
        
        # Check geometry validity (optional, requires shapely)
        invalid_geoms = geo_forecast[~geo_forecast['geometry'].is_valid]['precinct'].tolist()
        if invalid_geoms:
            logger.error(f"Invalid geometries found in precincts: {invalid_geoms}")
            return None
        
        # Check if there are any predicted calls
        total_predicted_calls = geo_forecast['predicted_calls'].sum()
        if total_predicted_calls == 0:
            logger.warning("No predicted calls available for mapping. Setting all demand to 'Low'.")
            geo_forecast['demand_category'] = 'Low'
            threshold = 0
        else:
            # Calculate threshold for demand categories
            positive_calls = geo_forecast['predicted_calls'][geo_forecast['predicted_calls'] > 0]
            threshold = np.percentile(positive_calls, 75) if len(positive_calls) > 0 else 0
            if threshold == 0:
                logger.warning("No positive predicted calls found. Using threshold of 0.")
            
            # Assign demand categories
            geo_forecast['demand_category'] = 'Low'
            geo_forecast.loc[geo_forecast['predicted_calls'] > threshold, 'demand_category'] = 'High'
        
        # Create choropleth map
        m = folium.Map(location=[40.7128, -74.0060], zoom_start=11)
        
        # Convert to GeoJSON for Choropleth (ensures compatibility)
        geo_json_data = geo_forecast.to_json()
        
        folium.Choropleth(
            geo_data=geo_json_data,
            name='Forecasted Call Volume',
            data=geo_forecast,
            columns=['precinct', 'predicted_calls'],
            key_on='feature.properties.precinct',  # Verify this matches your GeoJSON structure
            fill_color='YlOrRd',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name='Predicted Call Volume',
            highlight=True
        ).add_to(m)
        
        # Add tooltips
        folium.GeoJson(
            geo_json_data,
            name='Precinct Details',
            tooltip=folium.GeoJsonTooltip(
                fields=['precinct', 'predicted_calls', 'demand_category'],
                aliases=['Precinct:', 'Predicted Calls:', 'Demand Level:'],
                localize=True
            )
        ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save map
        m.save('eda_plots/forecasted_demand_map.html')
        logger.info("Geospatial prediction map created successfully")
        
        return geo_forecast
    
    except Exception as e:
        logger.error(f"Failed to create geospatial prediction: {e}")
        return None
    
def allocate_resources(geo_forecast, total_ambulances=100):
    """
    Allocate ambulances across precincts based on predicted call volumes.
    """
    logger.info("Allocating resources based on predicted demand...")
    
    # Calculate proportion of calls for each precinct
    total_predicted_calls = geo_forecast['predicted_calls'].sum()
    
    if total_predicted_calls == 0:
        logger.warning("No predicted calls found for resource allocation")
        geo_forecast['allocated_ambulances'] = 0
    else:
        geo_forecast['proportion'] = geo_forecast['predicted_calls'] / total_predicted_calls
        geo_forecast['allocated_ambulances'] = (geo_forecast['proportion'] * total_ambulances).apply(round)
    
    # Adjust to ensure we allocate exactly the target number of ambulances
    adjustment = total_ambulances - geo_forecast['allocated_ambulances'].sum()
    
    if adjustment != 0:
        # Add/subtract the difference to/from precincts with highest call volumes
        sorted_indices = geo_forecast['predicted_calls'].sort_values(ascending=False).index
        
        for i in range(abs(adjustment)):
            idx = sorted_indices[i % len(sorted_indices)]
            geo_forecast.loc[idx, 'allocated_ambulances'] += 1 if adjustment > 0 else -1
    
    # Prepare output
    resource_allocation = geo_forecast[['precinct', 'predicted_calls', 'allocated_ambulances']].copy()
    resource_allocation.sort_values('allocated_ambulances', ascending=False, inplace=True)
    resource_allocation.to_csv('data/resource_allocation.csv', index=False)
    
    logger.info(f"Allocated {total_ambulances} ambulances across {len(geo_forecast)} precincts")
    return resource_allocation

def print_insights(forecasted_calls, resource_allocation, anomalies):
    """
    Print key insights from the analysis.
    """
    logger.info("\n=== KEY INSIGHTS ===")
    
    # Top high-demand precincts
    top_precincts = resource_allocation.sort_values('allocated_ambulances', ascending=False).head(5)
    logger.info("\nTop 5 High-Demand Precincts:")
    for _, row in top_precincts.iterrows():
        logger.info(f"Precinct {row['precinct']}: {row['predicted_calls']:.0f} predicted calls, {row['allocated_ambulances']:.0f} ambulances allocated")
    
    # Anomalies summary
    if isinstance(anomalies, list) and anomalies:
        all_anomalies = pd.concat(anomalies)
        logger.info(f"\nDetected {len(all_anomalies)} anomalous days across precincts")
        logger.info(f"Average z-score of anomalies: {abs(all_anomalies['z_score']).mean():.2f}")
    else:
        logger.info("\nNo significant anomalies detected in historical data")
    
    # Overall forecast
    total_calls = forecasted_calls['predicted_calls'].sum()
    avg_daily = total_calls / forecasted_calls['date'].nunique()
    logger.info(f"\nTotal forecasted calls for next week: {total_calls:.0f}")
    logger.info(f"Average daily calls: {avg_daily:.0f}")
    
    logger.info("\n=== FORECASTING AND RESOURCE ALLOCATION COMPLETED ===")

def main():
    """
    Main function to run the entire forecasting and resource allocation workflow.
    """
    try:
        logger.info("=== STARTING NYPD EMERGENCY CALL FORECASTING ===")
        
        # Load data
        grouped_df, precinct_df = load_data()
        
        # Explore historical data
        hourly_agg, daily_agg = explore_historical_data(grouped_df)
        
        # Detect anomalies
        anomalies = detect_anomalies(grouped_df)
        
        # Forecast call volumes
        forecasted_calls = forecast_call_volumes(grouped_df)
        
        if forecasted_calls is None:
            logger.error("Forecasting failed. Exiting program.")
            return
        
        # Create geospatial prediction
        geo_forecast = create_geospatial_prediction(forecasted_calls, precinct_df)
        
        if geo_forecast is None:
            logger.error("Geospatial prediction failed. Exiting program.")
            return
        
        # Allocate resources
        resource_allocation = allocate_resources(geo_forecast, total_ambulances=100)
        
        # Print insights
        print_insights(forecasted_calls, resource_allocation, anomalies)
        
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()
