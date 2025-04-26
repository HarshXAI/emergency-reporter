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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('NYPD-Forecasting')

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
        
        # Load geospatial data
        precinct_df = gpd.read_file('data/merged_calls_precincts.csv')
        
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
    
    # Aggregate call counts by precinct, date, and hour
    hourly_agg = grouped_df.groupby(['date', 'hour']).sum()['call_count'].reset_index()
    daily_agg = grouped_df.groupby('date').sum()['call_count'].reset_index()
    
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
    call_types = grouped_df.groupby('TYP_DESC').sum()['call_count'].sort_values(ascending=False).head(5)
    sns.barplot(x=call_types.values, y=call_types.index)
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
            
            # Add precinct information to forecast
            forecast['PRECINCT'] = precinct
            
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
    """
    logger.info("Creating geospatial prediction map...")
    
    try:
        # Aggregate forecasted calls by precinct
        precinct_forecast = forecasted_calls.groupby('PRECINCT').sum()['predicted_calls'].reset_index()
        precinct_forecast.columns = ['precinct', 'predicted_calls']
        
        # Merge with geospatial data
        geo_forecast = precinct_df.merge(precinct_forecast, on='precinct', how='left')
        geo_forecast['predicted_calls'].fillna(0, inplace=True)
        
        # Identify high demand areas (top 25% of predicted calls)
        threshold = np.percentile(geo_forecast['predicted_calls'][geo_forecast['predicted_calls'] > 0], 75)
        geo_forecast['demand_category'] = 'Low'
        geo_forecast.loc[geo_forecast['predicted_calls'] > threshold, 'demand_category'] = 'High'
        
        # Create choropleth map
        m = folium.Map(location=[40.7128, -74.0060], zoom_start=11)
        
        # Add choropleth layer
        folium.Choropleth(
            geo_data=geo_forecast,
            name='Forecasted Call Volume',
            data=geo_forecast,
            columns=['precinct', 'predicted_calls'],
            key_on='feature.properties.precinct',
            fill_color='YlOrRd',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name='Predicted Call Volume',
            highlight=True
        ).add_to(m)
        
        # Add popup information
        folium.GeoJson(
            geo_forecast,
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
