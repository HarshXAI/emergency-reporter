import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import folium_static
from datetime import datetime, timedelta
import warnings
import os
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="NYPD Emergency Calls Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS to improve appearance and fix tab styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #1E3A8A;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .stat-box {
        background-color: #F0F7FF;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .chart-container {
        background-color: #FFFFFF;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    .st-emotion-cache-16txtl3 h4 {
        padding-top: 1rem;
        font-weight: bold;
        color: #1E3A8A;
    }
    /* Fixed tab styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        padding: 0;
        margin: 0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #e0eeff;
        border-radius: 6px 6px 0 0;
        padding: 10px 20px;
        font-weight: 500;
        margin: 0;
        color: #1E3A8A;  /* Ensure text is visible in non-active state */
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E3A8A !important;
        color: white !important;
    }
    /* Fix any text visibility issues in the tabs */
    button[role="tab"] {
        color: #1E3A8A;
    }
    button[role="tab"][aria-selected="true"] {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">NYPD Emergency Calls Analysis Dashboard</h1>', unsafe_allow_html=True)

# Cached data loading functions
@st.cache_data
def load_grouped_data():
    """Load and preprocess the grouped dataset"""
    try:
        df = pd.read_csv('data/grouped_data.csv')
        df['date'] = pd.to_datetime(df['date'])
        if 'PRECINCT' in df.columns:
            df['PRECINCT'] = df['PRECINCT'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        return df
    except FileNotFoundError:
        st.error("Could not find grouped data file. Please ensure data/grouped_data.csv exists.")
        return pd.DataFrame()

@st.cache_data
def load_precinct_geojson():
    """Load geospatial data for precincts"""
    try:
        gdf = gpd.read_file('data/merged_calls_precincts.geojson')
        gdf['precinct'] = gdf['precinct'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        return gdf
    except:
        try:
            gdf = gpd.read_file('/Users/harshkanani/Desktop/Spit/data/merged_calls_precincts.geojson')
            gdf['precinct'] = gdf['precinct'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            return gdf
        except:
            st.error("Could not load geospatial data for precincts.")
            return gpd.GeoDataFrame()

@st.cache_data
def load_anomalies():
    """Load anomaly detection results"""
    try:
        df = pd.read_csv('data/call_anomalies.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        st.warning("No anomalies data found.")
        return pd.DataFrame()

@st.cache_data
def load_forecast_data():
    """Load forecast data"""
    try:
        df = pd.read_csv('data/forecasted_calls_by_type.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        st.warning("No forecast data found.")
        return pd.DataFrame()

@st.cache_data
def load_resource_allocation():
    """Load resource allocation data"""
    try:
        return pd.read_csv('data/resource_allocation.csv')
    except FileNotFoundError:
        try:
            return pd.read_csv('data/resource_allocation_by_type.csv')
        except FileNotFoundError:
            st.warning("No resource allocation data found.")
            return pd.DataFrame()

@st.cache_data
def load_lstm_trend_data():
    """Load LSTM trend analysis data if available"""
    trend_files = list(Path("trend_analysis").glob("decomposition_*.csv"))
    if trend_files:
        trend_data = {}
        for file in trend_files:
            precinct_call = file.name.replace("decomposition_", "").replace(".csv", "")
            try:
                df = pd.read_csv(file)
                df['date'] = pd.to_datetime(df['date'])
                trend_data[precinct_call] = df
            except:
                pass
        return trend_data
    return {}

# Load all data at once
try:
    grouped_df = load_grouped_data()
    precinct_gdf = load_precinct_geojson()
    anomalies_df = load_anomalies()
    forecast_df = load_forecast_data()
    resource_df = load_resource_allocation()
    trend_data = load_lstm_trend_data()
    
    data_loaded = not grouped_df.empty
    
    if not data_loaded:
        st.error("⚠️ Error loading data. Please check that data files exist in the data directory.")
        st.stop()
except Exception as e:
    st.error(f"⚠️ Error loading data: {e}")
    st.stop()

# Calculate common data for reuse
if 'TYP_DESC' in grouped_df.columns:
    call_types = sorted(grouped_df['TYP_DESC'].unique())
precincts = sorted(grouped_df['PRECINCT'].unique())

# Create horizontal tabs
tab_overview, tab_historical, tab_anomalies, tab_forecast, tab_resources, tab_trends = st.tabs([
    "📊 Overview", 
    "📈 Historical Analysis", 
    "🔍 Anomaly Detection", 
    "🔮 Forecast Analysis", 
    "🚑 Resource Allocation", 
    "📉 Trend Analysis"
])

# Sidebar for filters only
with st.sidebar:
    st.title("Filters")
    
    min_date = grouped_df['date'].min().date()
    max_date = grouped_df['date'].max().date()
    date_range = st.date_input(
        "Select Date Range",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    selected_precinct = st.selectbox(
        "Select Precinct",
        ["All"] + precincts
    )
    
    selected_call_type = st.selectbox(
        "Select Call Type",
        ["All"] + call_types
    )

# Filter the data based on selections
if selected_precinct == "All":
    filtered_df = grouped_df[(grouped_df['date'].dt.date >= date_range[0]) & 
                             (grouped_df['date'].dt.date <= date_range[1])]
else:
    filtered_df = grouped_df[(grouped_df['PRECINCT'] == selected_precinct) & 
                             (grouped_df['date'].dt.date >= date_range[0]) & 
                             (grouped_df['date'].dt.date <= date_range[1])]

# OVERVIEW PAGE
with tab_overview:
    st.markdown('<h2 class="section-header">Emergency Calls Overview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="stat-box">', unsafe_allow_html=True)
        st.metric("Total Emergency Calls", f"{grouped_df['call_count'].sum():,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="stat-box">', unsafe_allow_html=True)
        st.metric("Total Precincts", f"{len(precincts):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="stat-box">', unsafe_allow_html=True)
        days_covered = (grouped_df['date'].max() - grouped_df['date'].min()).days + 1
        st.metric("Days Covered", f"{days_covered:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<h3>Call Volume Distribution by Precinct</h3>', unsafe_allow_html=True)
    
    if not precinct_gdf.empty:
        calls_by_precinct = grouped_df.groupby('PRECINCT')['call_count'].sum().reset_index()
        calls_by_precinct.columns = ['precinct', 'total_calls']
        
        calls_by_precinct['precinct'] = calls_by_precinct['precinct'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        geo_data = precinct_gdf.copy()
        geo_data['precinct'] = geo_data['precinct'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        geo_data = geo_data.merge(calls_by_precinct, on='precinct', how='left')
        geo_data['total_calls'] = geo_data['total_calls'].fillna(0)
        
        m = folium.Map(location=[40.7128, -74.0060], zoom_start=10, tiles='CartoDB positron')
        
        folium.Choropleth(
            geo_data=geo_data,
            name='Call Volume',
            data=geo_data,
            columns=['precinct', 'total_calls'],
            key_on='feature.properties.precinct',
            fill_color='YlOrRd',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name='Total Calls'
        ).add_to(m)
        
        folium.features.GeoJson(
            geo_data,
            name='Precinct Details',
            tooltip=folium.features.GeoJsonTooltip(
                fields=['precinct', 'total_calls'],
                aliases=['Precinct:', 'Total Calls:'],
                localize=True
            )
        ).add_to(m)
        
        folium.LayerControl().add_to(m)
        
        folium_static(m)
    else:
        st.warning("Geospatial data not available for map visualization.")
    
    st.markdown('<h3>Call Type Distribution</h3>', unsafe_allow_html=True)
    
    calls_by_type = grouped_df.groupby('TYP_DESC')['call_count'].sum().reset_index()
    calls_by_type = calls_by_type.sort_values('call_count', ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='call_count', y='TYP_DESC', data=calls_by_type, palette='viridis')
    plt.title('Top 10 Emergency Call Types', fontsize=16)
    plt.xlabel('Total Call Volume')
    plt.ylabel('Call Type')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown('<h3>Call Volume Trends Over Time</h3>', unsafe_allow_html=True)
    
    daily_calls = grouped_df.groupby(grouped_df['date'].dt.date)['call_count'].sum().reset_index()
    daily_calls['date'] = pd.to_datetime(daily_calls['date'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x='date', y='call_count', data=daily_calls)
    plt.title('Daily Call Volume', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Number of Calls')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

# HISTORICAL ANALYSIS PAGE
with tab_historical:
    st.markdown('<h2 class="section-header">Historical Data Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown('<h3>Daily Call Volume</h3>', unsafe_allow_html=True)
    
    daily_agg = filtered_df.groupby('date')['call_count'].sum().reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x='date', y='call_count', data=daily_agg)
    title = f'Daily Call Volume {"for Precinct "+selected_precinct if selected_precinct != "All" else "Across All Precincts"}'
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Total Call Count')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown('<h3>Hourly Call Volume Patterns</h3>', unsafe_allow_html=True)
    
    if 'hour' in filtered_df.columns:
        hourly_agg = filtered_df.groupby(['date', 'hour'])['call_count'].sum().reset_index()
        avg_hourly = hourly_agg.groupby('hour')['call_count'].mean().reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(x='hour', y='call_count', data=avg_hourly, marker='o')
        title = f'Average Call Volume by Hour {"for Precinct "+selected_precinct if selected_precinct != "All" else "Across All Precincts"}'
        plt.title(title)
        plt.xlabel('Hour of Day (24-hour format)')
        plt.ylabel('Average Call Count')
        plt.xticks(range(0, 24))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
    
    st.markdown('<h3>Call Volume by Day of Week</h3>', unsafe_allow_html=True)
    
    filtered_df['day_of_week'] = filtered_df['date'].dt.day_name()
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    day_agg = filtered_df.groupby('day_of_week')['call_count'].sum().reset_index()
    day_agg['day_of_week'] = pd.Categorical(day_agg['day_of_week'], categories=days_order, ordered=True)
    day_agg = day_agg.sort_values('day_of_week')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='day_of_week', y='call_count', data=day_agg)
    title = f'Total Call Volume by Day of Week {"for Precinct "+selected_precinct if selected_precinct != "All" else "Across All Precincts"}'
    plt.title(title)
    plt.xlabel('Day of Week')
    plt.ylabel('Total Call Count')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown('<h3>Top Call Types</h3>', unsafe_allow_html=True)
    
    type_agg = filtered_df.groupby('TYP_DESC')['call_count'].sum().reset_index().sort_values('call_count', ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='call_count', y='TYP_DESC', data=type_agg)
    title = f'Top 10 Call Types {"for Precinct "+selected_precinct if selected_precinct != "All" else "Across All Precincts"}'
    plt.title(title)
    plt.xlabel('Total Call Count')
    plt.ylabel('Call Type')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown('<h3>Data Explorer</h3>', unsafe_allow_html=True)
    
    if st.checkbox("Show Raw Data Sample"):
        st.dataframe(filtered_df.sort_values('date', ascending=False).head(1000))

# ANOMALY DETECTION PAGE
with tab_anomalies:
    st.markdown('<h2 class="section-header">Anomaly Detection</h2>', unsafe_allow_html=True)
    
    if not anomalies_df.empty:
        if selected_precinct == "All":
            filtered_anomalies = anomalies_df[(anomalies_df['date'].dt.date >= date_range[0]) & 
                                              (anomalies_df['date'].dt.date <= date_range[1])]
        else:
            filtered_anomalies = anomalies_df[(anomalies_df['PRECINCT'] == selected_precinct) & 
                                              (anomalies_df['date'].dt.date >= date_range[0]) & 
                                              (anomalies_df['date'].dt.date <= date_range[1])]
        
        st.markdown('<h3>Anomaly Statistics</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.metric("Total Anomalies Detected", f"{len(filtered_anomalies)}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            avg_zscore = filtered_anomalies['z_score'].abs().mean() if not filtered_anomalies.empty else 0
            st.metric("Average Anomaly Severity (|z-score|)", f"{avg_zscore:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        if not filtered_anomalies.empty:
            st.markdown('<h3>Anomaly Visualization</h3>', unsafe_allow_html=True)
            
            if selected_precinct == "All":
                normal_data = grouped_df.groupby('date')['call_count'].sum().reset_index()
            else:
                normal_data = grouped_df[grouped_df['PRECINCT'] == selected_precinct].groupby('date')['call_count'].sum().reset_index()
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            sns.lineplot(x='date', y='call_count', data=normal_data, label='Normal', alpha=0.7)
            
            sns.scatterplot(x='date', y='call_count', data=filtered_anomalies, 
                            color='red', s=80, label='Anomaly')
            
            title = f'Call Volume with Detected Anomalies {"for Precinct "+selected_precinct if selected_precinct != "All" else "Across All Precincts"}'
            plt.title(title)
            plt.xlabel('Date')
            plt.ylabel('Call Count')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown('<h3>Anomaly Details</h3>', unsafe_allow_html=True)
            
            display_anomalies = filtered_anomalies.copy()
            display_anomalies['abs_z_score'] = display_anomalies['z_score'].abs()
            display_anomalies = display_anomalies.sort_values('abs_z_score', ascending=False)
            
            st.dataframe(display_anomalies[['date', 'PRECINCT', 'call_count', 'z_score']])
            
            csv = display_anomalies.to_csv(index=False)
            st.download_button(
                label="Download Anomaly Data",
                data=csv,
                file_name='anomalies.csv',
                mime='text/csv',
            )
        else:
            st.info("No anomalies detected within the selected timeframe and precinct.")
    else:
        st.warning("Anomaly detection data is not available. Please run anomaly detection first.")

# FORECAST ANALYSIS PAGE
with tab_forecast:
    st.markdown('<h2 class="section-header">Forecast Analysis</h2>', unsafe_allow_html=True)
    
    if not forecast_df.empty:
        if selected_precinct == "All" and selected_call_type == "All":
            filtered_forecasts = forecast_df
        elif selected_precinct == "All":
            filtered_forecasts = forecast_df[forecast_df['TYP_DESC'] == selected_call_type]
        elif selected_call_type == "All":
            filtered_forecasts = forecast_df[forecast_df['PRECINCT'] == selected_precinct]
        else:
            filtered_forecasts = forecast_df[
                (forecast_df['PRECINCT'] == selected_precinct) & 
                (forecast_df['TYP_DESC'] == selected_call_type)
            ]
        
        st.markdown('<h3>Forecasted Call Volumes</h3>', unsafe_allow_html=True)
        
        if not filtered_forecasts.empty:
            if selected_precinct == "All" or selected_call_type == "All":
                agg_forecast = filtered_forecasts.groupby('date').agg({
                    'predicted_calls': 'sum',
                    'lower_bound': 'sum',
                    'upper_bound': 'sum'
                }).reset_index()
            else:
                agg_forecast = filtered_forecasts
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            plt.plot(agg_forecast['date'], agg_forecast['predicted_calls'], 'b-', 
                     label='Forecast')
            
            plt.fill_between(
                agg_forecast['date'],
                agg_forecast['lower_bound'],
                agg_forecast['upper_bound'],
                color='b', alpha=0.1,
                label='95% Confidence Interval'
            )
            
            title_parts = []
            if selected_precinct != "All":
                title_parts.append(f"Precinct {selected_precinct}")
            if selected_call_type != "All":
                title_parts.append(f"{selected_call_type}")
                
            title = f"Forecasted Call Volume for {' and '.join(title_parts)}" if title_parts else "Forecasted Call Volume Across All Precincts and Call Types"
            
            plt.title(title)
            plt.xlabel('Date')
            plt.ylabel('Predicted Call Count')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            st.pyplot(fig)
            
            if selected_precinct == "All" and not precinct_gdf.empty:
                st.markdown('<h3>Geographic Distribution of Forecasted Calls</h3>', unsafe_allow_html=True)
                
                precinct_forecasts = filtered_forecasts.groupby('PRECINCT')['predicted_calls'].sum().reset_index()
                
                precinct_forecasts.rename(columns={'PRECINCT': 'precinct'}, inplace=True)
                
                precinct_forecasts['precinct'] = precinct_forecasts['precinct'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
                geo_forecasts = precinct_gdf.copy()
                geo_forecasts['precinct'] = geo_forecasts['precinct'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
                
                geo_forecasts = geo_forecasts.merge(precinct_forecasts, on='precinct', how='left')
                geo_forecasts['predicted_calls'] = geo_forecasts['predicted_calls'].fillna(0)
                
                m = folium.Map(location=[40.7128, -74.0060], zoom_start=10, tiles='CartoDB positron')
                
                folium.Choropleth(
                    geo_data=geo_forecasts,
                    name='Forecasted Calls',
                    data=geo_forecasts,
                    columns=['precinct', 'predicted_calls'],
                    key_on='feature.properties.precinct',
                    fill_color='YlOrRd',
                    fill_opacity=0.7,
                    line_opacity=0.2,
                    legend_name='Forecasted Calls'
                ).add_to(m)
                
                folium.features.GeoJson(
                    geo_forecasts,
                    name='Precinct Details',
                    tooltip=folium.features.GeoJsonTooltip(
                        fields=['precinct', 'predicted_calls'],
                        aliases=['Precinct:', 'Forecasted Calls:'],
                        localize=True
                    )
                ).add_to(m)
                
                folium.LayerControl().add_to(m)
                
                folium_static(m)
                
                st.markdown("""
                <div style='font-style: italic; color: #666;'>
                The map shows the total predicted call volume for each precinct across the entire forecast period.
                Darker colors indicate higher forecasted call volumes.
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('<h3>Forecast Details</h3>', unsafe_allow_html=True)
            
            forecast_display = filtered_forecasts.copy()
            forecast_display['day_of_week'] = forecast_display['date'].dt.day_name()
            
            for col in ['predicted_calls', 'lower_bound', 'upper_bound']:
                if col in forecast_display.columns:
                    forecast_display[col] = forecast_display[col].round(0).astype(int)
            
            st.dataframe(forecast_display[['date', 'day_of_week', 'PRECINCT', 'TYP_DESC', 
                                           'predicted_calls', 'lower_bound', 'upper_bound']])
            
            csv = forecast_display.to_csv(index=False)
            st.download_button(
                label="Download Forecast Data",
                data=csv,
                file_name='forecasts.csv',
                mime='text/csv',
            )
        else:
            st.info("No forecast data available for the selected precinct and call type.")
    else:
        st.warning("Forecast data is not available. Please run forecasting first.")

# RESOURCE ALLOCATION PAGE
with tab_resources:
    st.markdown('<h2 class="section-header">Resource Allocation</h2>', unsafe_allow_html=True)
    
    if not resource_df.empty:
        st.markdown('<h3>Resource Allocation by Precinct</h3>', unsafe_allow_html=True)
        
        resource_columns = [col for col in resource_df.columns if col.startswith('allocated_')]
        
        if resource_columns:
            selected_resource = st.selectbox(
                "Select Resource Type",
                [col.replace('allocated_', '') for col in resource_columns]
            )
            
            resource_col = f'allocated_{selected_resource}'
            
            if resource_col in resource_df.columns:
                # Dynamically find the precinct column name
                precinct_col = next((col for col in resource_df.columns if 'precinct' in col.lower()), None)
                
                if precinct_col is None:
                    st.error("No precinct column found in resource allocation data. Available columns: " + 
                            ", ".join(resource_df.columns.tolist()))
                else:
                    allocation_display = resource_df.copy()
                    allocation_display = allocation_display.sort_values(resource_col, ascending=False)
                    top_precincts = allocation_display.head(20)
                    
                    if not top_precincts.empty:
                        fig, ax = plt.subplots(figsize=(12, 8))
                        sns.barplot(x=resource_col, y=precinct_col, data=top_precincts)
                        plt.title(f'Top 20 Precincts by {selected_resource.title()} Allocation')
                        plt.xlabel(f'Number of {selected_resource.title()}s Allocated')
                        plt.ylabel('Precinct')
                        plt.grid(axis='x', alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.info("No data available to display for the selected resource type.")
                    
                    if not precinct_gdf.empty:
                        st.markdown('<h3>Resource Allocation Map</h3>', unsafe_allow_html=True)
                        
                        # Ensure consistent column name for merging
                        allocation_map = allocation_display.copy()
                        allocation_map = allocation_map.rename(columns={precinct_col: 'precinct'}, inplace=False)
                        allocation_map['precinct'] = allocation_map['precinct'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
                        geo_precinct = precinct_gdf.copy()
                        geo_precinct['precinct'] = geo_precinct['precinct'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
                        
                        geo_allocation = geo_precinct.merge(allocation_map, on='precinct', how='left')
                        geo_allocation[resource_col] = geo_allocation[resource_col].fillna(0)
                        
                        m = folium.Map(location=[40.7128, -74.0060], zoom_start=10, tiles='CartoDB positron')
                        
                        folium.Choropleth(
                            geo_data=geo_allocation,
                            name=f'{selected_resource.title()} Allocation',
                            data=geo_allocation,
                            columns=['precinct', resource_col],
                            key_on='feature.properties.precinct',
                            fill_color='YlGnBu',
                            fill_opacity=0.7,
                            line_opacity=0.2,
                            legend_name=f'{selected_resource.title()} Allocation'
                        ).add_to(m)
                        
                        folium.features.GeoJson(
                            geo_allocation,
                            name='Precinct Details',
                            tooltip=folium.features.GeoJsonTooltip(
                                fields=['precinct', resource_col],
                                aliases=['Precinct:', f'{selected_resource.title()}s:'],
                                localize=True
                            )
                        ).add_to(m)
                        
                        folium.LayerControl().add_to(m)
                        
                        folium_static(m)
                    
                    st.markdown('<h3>Resource Allocation Details</h3>', unsafe_allow_html=True)
                    st.dataframe(allocation_display)
                    
                    csv = allocation_display.to_csv(index=False)
                    st.download_button(
                        label="Download Resource Allocation Data",
                        data=csv,
                        file_name='resource_allocation.csv',
                        mime='text/csv',
                    )
            else:
                st.info(f"No allocation data available for {selected_resource}.")
        else:
            st.info("No resource allocation details found in the data.")
    else:
        st.warning("Resource allocation data is not available. Please run resource allocation first.")

# TREND ANALYSIS PAGE
with tab_trends:
    st.markdown('<h2 class="section-header">Trend Analysis with LSTM</h2>', unsafe_allow_html=True)
    
    if trend_data:
        trend_keys = list(trend_data.keys())
        selected_trend = st.selectbox("Select Precinct and Call Type", trend_keys)
        
        if selected_trend in trend_data:
            decomposition_df = trend_data[selected_trend]
            
            st.markdown('<h3>Decomposed Time Series</h3>', unsafe_allow_html=True)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            ax1.plot(decomposition_df['date'], decomposition_df['original'], label='Original', alpha=0.7)
            ax1.plot(decomposition_df['date'], decomposition_df['trend'], label='Long-term Trend', linewidth=2)
            ax1.set_title(f'Long-term Trend Analysis for {selected_trend}')
            ax1.set_ylabel('Call Count')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(decomposition_df['date'], decomposition_df['fluctuation'], label='Short-term Fluctuations', alpha=0.7)
            
            if 'z_score' in decomposition_df.columns:
                anomalies = decomposition_df[abs(decomposition_df['z_score']) > 2]
                if not anomalies.empty:
                    ax2.scatter(anomalies['date'], anomalies['fluctuation'], color='red', s=50, label='Anomalies')
            
            ax2.set_title('Short-term Fluctuations and Anomalies')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Fluctuation')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown('<h3>Trend Analysis Statistics</h3>', unsafe_allow_html=True)
            
            trend_slope = np.polyfit(range(len(decomposition_df)), decomposition_df['trend'], 1)[0]
            trend_direction = "Increasing" if trend_slope > 0 else "Decreasing"
            trend_mean = decomposition_df['trend'].mean()
            fluct_std = decomposition_df['fluctuation'].std()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="stat-box">', unsafe_allow_html=True)
                st.metric("Trend Direction", trend_direction)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col2:
                st.markdown('<div class="stat-box">', unsafe_allow_html=True)
                st.metric("Trend Slope", f"{trend_slope:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col3:
                st.markdown('<div class="stat-box">', unsafe_allow_html=True)
                st.metric("Fluctuation Volatility (STD)", f"{fluct_std:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<h3>Decomposition Data</h3>', unsafe_allow_html=True)
            st.dataframe(decomposition_df)
            
            csv = decomposition_df.to_csv(index=False)
            st.download_button(
                label="Download Decomposition Data",
                data=csv,
                file_name=f'trend_decomposition_{selected_trend}.csv',
                mime='text/csv',
            )
    else:
        st.warning("""LSTM trend analysis data is not available. 
        Run the lstm_trend_analysis.py script first to generate trend decomposition data.""")
        
        st.markdown('### Example of Trend Analysis')
        st.image('https://matplotlib.org/3.5.1/_images/sphx_glr_plot_arima_001.png', 
                 caption='Example of time series decomposition showing trend, seasonality, and residuals.')

# Add a footer
st.markdown("""---""")
st.markdown(
    """<div style='text-align: center; color: #666;'>
    NYPD Emergency Calls Analysis Dashboard | Data Science Team | Last updated: 
    </div>""" + 
    f"""<div style='text-align: center; color: #666;'>{datetime.now().strftime('%B %d, %Y')}</div>""",
    unsafe_allow_html=True
)