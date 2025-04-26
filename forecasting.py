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

# Create output directories if they don’t exist
Path("data").mkdir(exist_ok=True)
Path("eda_plots").mkdir(exist_ok=True)

# Define resource types and their corresponding call types
RESOURCE_MAPPING = {
    'ambulance': ['MEDICAL', 'INJURY', 'ILLNESS'],
    'fire_truck': ['FIRE', 'SMOKE', 'EXPLOSION'],
    'police_unit': ['CRIME', 'DISPUTE', 'THEFT']
}

def load_data():
    """
    Load the grouped call data and merged calls-precincts data.
    """
    try:
        logger.info("Loading datasets...")
        grouped_df = pd.read_csv('data/grouped_data.csv')
        grouped_df['date'] = pd.to_datetime(grouped_df['date'])
        
        if 'PRECINCT' in grouped_df.columns:
            grouped_df['PRECINCT'] = grouped_df['PRECINCT'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            logger.info(f"Unique PRECINCT values in grouped_df: {sorted(grouped_df['PRECINCT'].unique().tolist())}")
        
        precinct_df = gpd.read_file('/Users/harshkanani/Desktop/Spit/data/merged_calls_precincts.geojson')
        
        if 'precinct' in precinct_df.columns:
            precinct_df['precinct'] = precinct_df['precinct'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            logger.info(f"Unique precinct values in precinct_df: {sorted(precinct_df['precinct'].unique().tolist())}")
        else:
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
    
    logger.info(f"Columns in dataframe: {grouped_df.columns.tolist()}")
    logger.info(f"Data types: {grouped_df.dtypes}")
    
    grouped_df['date'] = pd.to_datetime(grouped_df['date'])
    grouped_df['hour'] = pd.to_numeric(grouped_df['hour'], errors='coerce')
    
    hourly_agg = grouped_df.groupby(['date', 'hour']).agg({'call_count': 'sum'}).reset_index()
    daily_agg = grouped_df.groupby('date').agg({'call_count': 'sum'}).reset_index()
    
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
    
    daily_precinct = grouped_df.groupby(['date', 'PRECINCT']).sum()['call_count'].reset_index()
    anomalies = []
    
    for precinct in daily_precinct['PRECINCT'].unique():
        precinct_data = daily_precinct[daily_precinct['PRECINCT'] == precinct]
        mean_calls = precinct_data['call_count'].mean()
        std_calls = precinct_data['call_count'].std()
        
        if std_calls == 0:
            continue
            
        precinct_data['z_score'] = (precinct_data['call_count'] - mean_calls) / std_calls
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

def forecast_call_volumes_by_type(grouped_df):
    """
    Forecast call volumes for each precinct and call type using Prophet.
    """
    logger.info("Forecasting call volumes by call type...")
    
    daily_precinct_type = grouped_df.groupby(['date', 'PRECINCT', 'TYP_DESC']).sum()['call_count'].reset_index()
    daily_precinct_type['ds'] = daily_precinct_type['date']
    daily_precinct_type['y'] = daily_precinct_type['call_count']
    
    all_forecasts = []
    precincts = daily_precinct_type['PRECINCT'].unique()
    call_types = daily_precinct_type['TYP_DESC'].unique()
    
    for precinct in precincts:
        for call_type in call_types:
            try:
                logger.info(f"Forecasting for Precinct {precinct}, Call Type {call_type}...")
                precinct_type_data = daily_precinct_type[
                    (daily_precinct_type['PRECINCT'] == precinct) & 
                    (daily_precinct_type['TYP_DESC'] == call_type)
                ]
                
                if precinct_type_data.empty:
                    continue
                
                model = Prophet(
                    daily_seasonality=True,
                    weekly_seasonality=True,
                    yearly_seasonality=False,
                    interval_width=0.95,
                    seasonality_mode='multiplicative'
                )
                model.fit(precinct_type_data[['ds', 'y']])
                
                future = model.make_future_dataframe(periods=7, freq='D')
                forecast = model.predict(future)
                
                forecast['PRECINCT'] = str(precinct).replace('.0', '').strip()
                forecast['TYP_DESC'] = call_type
                
                max_date = precinct_type_data['ds'].max()
                forecast = forecast[forecast['ds'] > max_date]
                
                forecast_subset = forecast[['ds', 'PRECINCT', 'TYP_DESC', 'yhat', 'yhat_lower', 'yhat_upper']]
                forecast_subset.columns = ['date', 'PRECINCT', 'TYP_DESC', 'predicted_calls', 'lower_bound', 'upper_bound']
                all_forecasts.append(forecast_subset)
                
            except Exception as e:
                logger.warning(f"Forecasting failed for Precinct {precinct}, Call Type {call_type}: {e}")
                continue
    
    if all_forecasts:
        forecasted_calls = pd.concat(all_forecasts)
        forecasted_calls['predicted_calls'] = forecasted_calls['predicted_calls'].apply(lambda x: max(0, round(x)))
        forecasted_calls['lower_bound'] = forecasted_calls['lower_bound'].apply(lambda x: max(0, round(x)))
        forecasted_calls['upper_bound'] = forecasted_calls['upper_bound'].apply(lambda x: max(0, round(x)))
        
        forecasted_calls.to_csv('data/forecasted_calls_by_type.csv', index=False)
        logger.info(f"Generated forecasts for {len(precincts)} precincts and {len(call_types)} call types")
        return forecasted_calls
    else:
        logger.error("Failed to generate forecasts for any precinct and call type")
        return None

def allocate_resources_by_type(forecasted_calls, resource_types, total_resources):
    """
    Allocate resources across precincts based on predicted call volumes for each resource type.
    """
    logger.info("Allocating resources based on predicted demand by type...")
    
    resource_allocation = {}
    
    for resource, call_types in resource_types.items():
        logger.info(f"Allocating {resource}...")
        
        # Filter calls relevant to the current resource type
        relevant_calls = forecasted_calls[forecasted_calls['TYP_DESC'].isin(call_types)]
        
        # Sum only the 'predicted_calls' column for each precinct
        precinct_calls = relevant_calls.groupby('PRECINCT')['predicted_calls'].sum().reset_index()
        
        total_predicted_calls = precinct_calls['predicted_calls'].sum()
        
        if total_predicted_calls == 0:
            logger.warning(f"No predicted calls for {resource}")
            precinct_calls['allocated_' + resource] = 0
        else:
            precinct_calls['proportion'] = precinct_calls['predicted_calls'] / total_predicted_calls
            precinct_calls['allocated_' + resource] = (precinct_calls['proportion'] * total_resources[resource]).apply(round)
        
        # Adjust allocation to match total resources
        adjustment = total_resources[resource] - precinct_calls['allocated_' + resource].sum()
        
        if adjustment != 0 and len(precinct_calls) > 0:  # Check if there are any precincts to adjust
            sorted_indices = precinct_calls['predicted_calls'].sort_values(ascending=False).index
            if len(sorted_indices) > 0:  # Additional check to ensure sorted_indices is not empty
                for i in range(abs(adjustment)):
                    idx = sorted_indices[i % len(sorted_indices)]
                    precinct_calls.loc[idx, 'allocated_' + resource] += 1 if adjustment > 0 else -1
            else:
                logger.warning(f"Cannot adjust {resource} allocation: no precincts with predicted calls")
        
        resource_allocation[resource] = precinct_calls[['PRECINCT', 'allocated_' + resource]]
    
    # Combine allocations for all resource types
    if 'ambulance' in resource_allocation and not resource_allocation['ambulance'].empty:
        final_allocation = resource_allocation['ambulance']
        for resource in resource_types.keys():
            if resource != 'ambulance' and resource in resource_allocation:
                final_allocation = final_allocation.merge(
                    resource_allocation[resource], on='PRECINCT', how='left'
                )
    else:
        # Create an empty DataFrame with the correct columns if no allocations exist
        columns = ['PRECINCT'] + [f'allocated_{resource}' for resource in resource_types.keys()]
        final_allocation = pd.DataFrame(columns=columns)
    
    final_allocation.fillna(0, inplace=True)  # Fill NaN with 0 for precincts with no allocation
    final_allocation.to_csv('data/resource_allocation_by_type.csv', index=False)
    
    logger.info(f"Allocated resources for {len(resource_types)} types across {len(final_allocation)} precincts")
    return final_allocation

def main():
    """
    Main function to run the forecasting and resource allocation workflow.
    """
    try:
        logger.info("=== STARTING NYPD EMERGENCY CALL FORECASTING ===")
        
        grouped_df, precinct_df = load_data()
        hourly_agg, daily_agg = explore_historical_data(grouped_df)
        anomalies = detect_anomalies(grouped_df)
        
        forecasted_calls_by_type = forecast_call_volumes_by_type(grouped_df)
        
        if forecasted_calls_by_type is None:
            logger.error("Forecasting failed. Exiting program.")
            return
        
        total_resources = {
            'ambulance': 100,
            'fire_truck': 50,
            'police_unit': 200
        }
        resource_allocation_by_type = allocate_resources_by_type(
            forecasted_calls_by_type, RESOURCE_MAPPING, total_resources
        )
        
        logger.info("\n=== RESOURCE ALLOCATION SUMMARY ===")
        logger.info(resource_allocation_by_type.head())
        
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()