import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger("NYPD-LSTM-Trend-Analysis")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Create output directories if they don't exist
Path("data").mkdir(exist_ok=True)
Path("trend_analysis").mkdir(exist_ok=True)

def load_data():
    """
    Load the grouped call data.
    """
    try:
        logger.info("Loading dataset...")
        # Load time-series data
        grouped_df = pd.read_csv('data/grouped_data.csv')
        # Convert date to datetime
        grouped_df['date'] = pd.to_datetime(grouped_df['date'])
        
        # Standardize PRECINCT column
        if 'PRECINCT' in grouped_df.columns:
            grouped_df['PRECINCT'] = grouped_df['PRECINCT'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        
        logger.info("Data loaded successfully")
        return grouped_df
    except FileNotFoundError as e:
        logger.error(f"Failed to load data file: {e}")
        raise

def prepare_time_series(grouped_df, precinct=None, call_type=None):
    """
    Prepare time series data for a specific precinct and call type.
    
    Parameters:
    - grouped_df: DataFrame containing emergency call data
    - precinct: Precinct to analyze (None for all precincts)
    - call_type: Call type to analyze (None for all types)
    
    Returns:
    - DataFrame with continuous date range and call counts
    """
    # Filter by precinct and call type if specified
    if precinct is not None:
        grouped_df = grouped_df[grouped_df['PRECINCT'] == precinct]
    if call_type is not None:
        grouped_df = grouped_df[grouped_df['TYP_DESC'] == call_type]
    
    # Aggregate by date
    daily_series = grouped_df.groupby('date')['call_count'].sum().reset_index()
    
    # Create continuous date range and fill missing dates with 0
    date_range = pd.date_range(start=daily_series['date'].min(), end=daily_series['date'].max())
    complete_series = pd.DataFrame({'date': date_range})
    complete_series = complete_series.merge(daily_series, on='date', how='left')
    complete_series['call_count'] = complete_series['call_count'].fillna(0)
    
    return complete_series

def create_lstm_trend_model(seq_length=30, trend_units=64, fluct_units=32):
    """
    Create an LSTM model that separates long-term trends from short-term fluctuations.
    
    Parameters:
    - seq_length: Number of previous days to use as input
    - trend_units: Number of units in the trend LSTM layer
    - fluct_units: Number of units in the fluctuation LSTM layer
    
    Returns:
    - trend_model: Model that extracts long-term trends
    - fluctuation_model: Model that extracts short-term fluctuations
    """
    # Trend model - captures long-term patterns with deeper architecture
    trend_model = Sequential([
        LSTM(trend_units, activation='tanh', return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(int(trend_units/2), activation='tanh'),
        Dropout(0.2),
        Dense(1)
    ])
    
    trend_model.compile(optimizer='adam', loss='mse')
    
    # Fluctuation model - captures short-term variations with simpler architecture
    fluctuation_model = Sequential([
        LSTM(fluct_units, activation='relu', input_shape=(seq_length, 1)),
        Dense(1)
    ])
    
    fluctuation_model.compile(optimizer='adam', loss='mse')
    
    return trend_model, fluctuation_model

def create_sequences(data, seq_length):
    """
    Create sequence inputs and targets for LSTM model.
    
    Parameters:
    - data: Time series data
    - seq_length: Number of previous time steps to use
    
    Returns:
    - X: Input sequences
    - y: Target values
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def train_trend_decomposition_models(time_series, seq_length=30, epochs=50, batch_size=32):
    """
    Train LSTM models to decompose a time series into trend and fluctuation components.
    
    Parameters:
    - time_series: DataFrame with 'date' and 'call_count' columns
    - seq_length: Number of previous days to consider
    - epochs: Number of training epochs
    - batch_size: Batch size for training
    
    Returns:
    - trend_model: Trained trend model
    - fluctuation_model: Trained fluctuation model
    - scaler: Fitted scaler for data transformation
    """
    # Extract and scale the call counts
    series = time_series['call_count'].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    series_scaled = scaler.fit_transform(series.reshape(-1, 1))
    
    # Create sequences
    X, y = create_sequences(series_scaled, seq_length)
    
    # Split into training and validation sets (80/20)
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # Create models
    trend_model, fluctuation_model = create_lstm_trend_model(seq_length)
    
    # Set up callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train trend model
    logger.info("Training long-term trend model...")
    trend_model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Generate trend predictions
    trend_preds_scaled = np.concatenate([
        np.zeros((seq_length, 1)),  # Placeholder for initial sequence
        trend_model.predict(X)
    ])
    
    # Prepare input for fluctuation model (original - trend)
    residuals_scaled = series_scaled - trend_preds_scaled
    
    # Create sequences for residuals
    X_res, y_res = create_sequences(residuals_scaled, seq_length)
    
    # Split residual data
    X_res_train, X_res_val = X_res[:train_size], X_res[train_size:]
    y_res_train, y_res_val = y_res[:train_size], y_res[train_size:]
    
    # Train fluctuation model on residuals
    logger.info("Training short-term fluctuation model...")
    fluctuation_model.fit(
        X_res_train, y_res_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_res_val, y_res_val),
        callbacks=[early_stopping],
        verbose=1
    )
    
    return trend_model, fluctuation_model, scaler

def decompose_time_series(time_series, trend_model, fluctuation_model, scaler, seq_length=30):
    """
    Decompose a time series into trend and fluctuation components.
    
    Parameters:
    - time_series: DataFrame with 'date' and 'call_count' columns
    - trend_model: Trained trend model
    - fluctuation_model: Trained fluctuation model
    - scaler: Fitted scaler
    - seq_length: Sequence length used for model training
    
    Returns:
    - result_df: DataFrame with original, trend, and fluctuation components
    """
    # Extract and scale the call counts
    series = time_series['call_count'].values
    series_scaled = scaler.transform(series.reshape(-1, 1))
    
    # Create sequences for the entire series
    X_full = []
    for i in range(len(series) - seq_length):
        X_full.append(series_scaled[i:i + seq_length])
    X_full = np.array(X_full)
    
    # Generate trend predictions
    trend_pred_scaled = np.zeros_like(series_scaled)
    trend_pred_scaled[seq_length:] = trend_model.predict(X_full).flatten().reshape(-1, 1)
    
    # Use moving average for the first seq_length values
    for i in range(seq_length):
        trend_pred_scaled[i] = np.mean(series_scaled[max(0, i-5):min(i+6, seq_length)])
    
    # Calculate residuals
    residuals_scaled = series_scaled - trend_pred_scaled
    
    # Create sequences for residuals
    X_res_full = []
    for i in range(len(residuals_scaled) - seq_length):
        X_res_full.append(residuals_scaled[i:i + seq_length])
    X_res_full = np.array(X_res_full)
    
    # Generate fluctuation predictions
    fluct_pred_scaled = np.zeros_like(series_scaled)
    fluct_pred_scaled[seq_length:] = fluctuation_model.predict(X_res_full).flatten().reshape(-1, 1)
    
    # Use residuals for the first seq_length values
    fluct_pred_scaled[:seq_length] = residuals_scaled[:seq_length]
    
    # Inverse transform to get original scale
    trend_pred = scaler.inverse_transform(trend_pred_scaled)
    fluct_pred = scaler.inverse_transform(series_scaled) - scaler.inverse_transform(trend_pred_scaled)
    
    # Create result DataFrame
    result_df = pd.DataFrame({
        'date': time_series['date'],
        'original': time_series['call_count'],
        'trend': trend_pred.flatten(),
        'fluctuation': fluct_pred.flatten()
    })
    
    return result_df

def detect_anomalies_in_fluctuations(decomposition_df, z_threshold=3.0):
    """
    Detect anomalies in the short-term fluctuations using z-score.
    
    Parameters:
    - decomposition_df: DataFrame with decomposed time series
    - z_threshold: Z-score threshold for anomaly detection
    
    Returns:
    - anomalies_df: DataFrame with detected anomalies
    """
    # Calculate z-scores for fluctuations
    mean_fluct = decomposition_df['fluctuation'].mean()
    std_fluct = decomposition_df['fluctuation'].std()
    
    if std_fluct == 0:  # Avoid division by zero
        decomposition_df['z_score'] = 0
    else:
        decomposition_df['z_score'] = (decomposition_df['fluctuation'] - mean_fluct) / std_fluct
    
    # Detect anomalies
    anomalies_df = decomposition_df[abs(decomposition_df['z_score']) > z_threshold].copy()
    
    return anomalies_df

def analyze_precinct_trend_patterns(grouped_df, precinct, call_types=None, seq_length=30):
    """
    Analyze trend patterns for a specific precinct across multiple call types.
    
    Parameters:
    - grouped_df: DataFrame with emergency call data
    - precinct: Precinct to analyze
    - call_types: List of call types to analyze (None for all types)
    - seq_length: Sequence length for LSTM models
    
    Returns:
    - trend_results: Dictionary with trend analysis results
    """
    logger.info(f"Analyzing trends for Precinct {precinct}")
    
    if call_types is None:
        call_types = grouped_df[grouped_df['PRECINCT'] == precinct]['TYP_DESC'].unique()
    
    trend_results = {}
    
    for call_type in call_types:
        try:
            logger.info(f"Analyzing trends for call type: {call_type}")
            
            # Prepare time series data
            time_series = prepare_time_series(grouped_df, precinct=precinct, call_type=call_type)
            
            if len(time_series) < seq_length * 2:
                logger.warning(f"Not enough data points for call type {call_type}. Skipping.")
                continue
            
            # Train decomposition models
            trend_model, fluctuation_model, scaler = train_trend_decomposition_models(time_series, seq_length=seq_length)
            
            # Apply decomposition
            decomposition_df = decompose_time_series(time_series, trend_model, fluctuation_model, scaler, seq_length)
            
            # Detect anomalies in fluctuations
            anomalies_df = detect_anomalies_in_fluctuations(decomposition_df)
            
            # Calculate trend metrics
            trend_metrics = {
                'trend_slope': np.polyfit(range(len(decomposition_df)), decomposition_df['trend'], 1)[0],
                'trend_mean': decomposition_df['trend'].mean(),
                'fluct_std': decomposition_df['fluctuation'].std(),
                'anomaly_count': len(anomalies_df),
                'max_anomaly_z_score': anomalies_df['z_score'].abs().max() if not anomalies_df.empty else 0
            }
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            
            # Plot original and trend
            ax1.plot(decomposition_df['date'], decomposition_df['original'], label='Original', alpha=0.7)
            ax1.plot(decomposition_df['date'], decomposition_df['trend'], label='Long-term Trend', linewidth=2)
            ax1.set_title(f'Long-term Trend Analysis - Precinct {precinct}, {call_type}')
            ax1.set_ylabel('Call Count')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot fluctuations and anomalies
            ax2.plot(decomposition_df['date'], decomposition_df['fluctuation'], label='Short-term Fluctuations', alpha=0.7)
            if not anomalies_df.empty:
                ax2.scatter(anomalies_df['date'], anomalies_df['fluctuation'], color='red', s=50, label='Anomalies')
            ax2.set_title('Short-term Fluctuations and Anomalies')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Fluctuation')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'trend_analysis/precinct_{precinct}_{call_type.replace(" ", "_")}.png')
            plt.close()
            
            # Save decomposition data
            decomposition_df.to_csv(f'trend_analysis/decomposition_{precinct}_{call_type.replace(" ", "_")}.csv', index=False)
            
            # Store results
            trend_results[call_type] = {
                'decomposition_df': decomposition_df,
                'anomalies_df': anomalies_df,
                'metrics': trend_metrics
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trends for call type {call_type}: {e}")
            continue
    
    return trend_results

def generate_trend_summary(trend_results):
    """
    Generate a summary of trend analysis results.
    
    Parameters:
    - trend_results: Dictionary with trend analysis results
    
    Returns:
    - summary_df: DataFrame with trend summary metrics
    """
    summary_data = []
    
    for call_type, results in trend_results.items():
        metrics = results['metrics']
        
        summary_data.append({
            'call_type': call_type,
            'trend_direction': 'Increasing' if metrics['trend_slope'] > 0 else 'Decreasing',
            'trend_slope': metrics['trend_slope'],
            'trend_mean': metrics['trend_mean'],
            'fluctuation_volatility': metrics['fluct_std'],
            'anomaly_count': metrics['anomaly_count'],
            'max_anomaly_severity': metrics['max_anomaly_z_score']
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Sort by trend slope to highlight fastest growing/declining call types
    summary_df = summary_df.sort_values('trend_slope', ascending=False)
    
    return summary_df

def main():
    """
    Main function for LSTM-based trend decomposition analysis.
    """
    try:
        logger.info("=== STARTING NYPD EMERGENCY CALL TREND ANALYSIS WITH LSTM ===")
        
        # Load data
        grouped_df = load_data()
        
        # Select a precinct for analysis (can be specified as a parameter)
        # Filter for precincts with sufficient data
        precinct_counts = grouped_df.groupby('PRECINCT')['call_count'].count()
        valid_precincts = precinct_counts[precinct_counts > 60].index.tolist()
        
        if not valid_precincts:
            logger.error("No precincts with sufficient data found")
            return
        
        # Get the top 5 precincts by call volume
        top_precincts = grouped_df.groupby('PRECINCT')['call_count'].sum().sort_values(ascending=False).head(5).index.tolist()
        
        # For each top precinct, get the top 3 call types
        for precinct in top_precincts:
            top_call_types = (grouped_df[grouped_df['PRECINCT'] == precinct]
                             .groupby('TYP_DESC')['call_count']
                             .sum()
                             .sort_values(ascending=False)
                             .head(3)
                             .index.tolist())
            
            # Analyze trends
            trend_results = analyze_precinct_trend_patterns(grouped_df, precinct, call_types=top_call_types)
            
            # Generate summary
            if trend_results:
                summary_df = generate_trend_summary(trend_results)
                summary_df.to_csv(f'trend_analysis/trend_summary_precinct_{precinct}.csv', index=False)
                
                logger.info(f"Trend analysis for Precinct {precinct} completed")
                logger.info(f"Summary:\n{summary_df}")
            else:
                logger.warning(f"No trend results for Precinct {precinct}")
        
        logger.info("=== LSTM TREND ANALYSIS COMPLETED ===")
        
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()
