# NYPD Emergency Call Analysis and Resource Allocation

## Project Overview

This project analyzes NYPD emergency call data to identify trends, forecast future call volumes, and optimize resource allocation across precincts. The system uses advanced time series analysis and machine learning techniques to help emergency services deploy resources more effectively.

## Components

The project consists of two main components:

1. **LSTM Trend Analysis** (`lstm_trend_analysis.py`): Decomposes emergency call data into long-term trends and short-term fluctuations using LSTM neural networks, helping identify underlying patterns and anomalies.

2. **Forecasting and Resource Allocation** (`forecasting.py`): Predicts future emergency call volumes by precinct and call type using Prophet, and then allocates emergency resources based on predicted demand.

## Datasets

The project uses the following data sources:

- **grouped_data.csv**: Time series data of emergency calls aggregated by date, hour, precinct, and call type. Each record includes:
  - `date`: Date of the calls
  - `hour`: Hour of the day (0-23)
  - `PRECINCT`: NYPD precinct number
  - `TYP_DESC`: Description of the emergency call type
  - `call_count`: Number of calls for that combination of factors

- **merged_calls_precincts.geojson**: Geospatial data that maps emergency calls to precinct boundaries, enabling geographical analysis and visualization.

## Methodology

### Time Series Decomposition with LSTMs

The LSTM trend analysis decomposes call volume time series into:

1. **Long-term trends**: Underlying patterns that evolve slowly over time, representing structural changes in call volumes
2. **Short-term fluctuations**: Temporary variations that may represent anomalous events or seasonal effects

This decomposition helps identify:
- Areas with consistently increasing or decreasing call volumes
- Anomalous spikes that may require investigation
- Seasonal patterns that can inform staffing decisions

### Forecasting with Prophet

The forecasting component uses Facebook's Prophet model to predict future call volumes because:
- It handles seasonal patterns effectively
- It's robust to missing data and outliers
- It can incorporate holidays and special events
- It provides uncertainty estimates for forecasts

### Resource Allocation

Resources (ambulances, fire trucks, police units) are allocated based on:
1. Call type mapping to resource needs
2. Forecasted demand by precinct
3. Proportional allocation with adjustments for rounding

## Features and Benefits

1. **Trend Detection**: Identifies precincts with consistently increasing emergency call volumes, allowing for proactive resource planning.

2. **Anomaly Detection**: Flags unusual spikes in call volumes that may indicate emerging situations requiring investigation.

3. **Call Type Analysis**: Breaks down emergency calls by type to identify the specific resources needed in each area.

4. **Demand Forecasting**: Predicts call volumes 7 days in advance to support staffing and resource planning.

5. **Optimized Resource Allocation**: Distributes limited resources across precincts based on predicted demand to maximize emergency response effectiveness.

6. **Visualization**: Generates plots and maps for intuitive understanding of patterns and trends.

## How to Use

1. Ensure input data files (`grouped_data.csv` and geospatial data) are available in the `data` directory.

2. Run the LSTM trend analysis:
   ```
   python lstm_trend_analysis.py
   ```

3. Run the forecasting and resource allocation:
   ```
   python forecasting.py
   ```

4. Review output files in:
   - `trend_analysis/`: Decomposition results and anomaly detection
   - `data/`: Forecast results and resource allocation plans
   - `eda_plots/`: Exploratory data analysis visualizations

## Technologies Used

- **TensorFlow/Keras**: For LSTM modeling of time series data
- **Prophet**: For robust time series forecasting
- **Pandas/NumPy**: For data manipulation and analysis
- **Matplotlib/Seaborn**: For data visualization
- **GeoPandas/Folium**: For geospatial analysis and mapping

## Future Enhancements

- Integration with real-time call data for dynamic resource reallocation
- Incorporation of external factors (weather, events, etc.) for improved forecasting
- Interactive dashboard for monitoring and decision support
- Optimization algorithms for resource routing and dispatch
