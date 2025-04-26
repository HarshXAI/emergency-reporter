import pandas as pd
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_timestamps(df):
    """Parse INCIDENT_DATE and INCIDENT_TIME into a datetime timestamp column."""
    result_df = df.copy()
    
    # Verify required columns
    required_cols = ['INCIDENT_DATE', 'INCIDENT_TIME']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.error(f"Missing columns: {missing_cols}")
        raise ValueError(f"Missing columns: {missing_cols}")
    
    # Combine date and time into a single string
    datetime_str = result_df['INCIDENT_DATE'] + ' ' + result_df['INCIDENT_TIME']
    
    # Convert to datetime, coercing errors to NaT
    result_df['timestamp'] = pd.to_datetime(datetime_str, format='%m/%d/%Y %H:%M:%S', errors='coerce')
    
    # Log parsing results
    valid_count = result_df['timestamp'].notna().sum()
    invalid_count = result_df['timestamp'].isna().sum()
    logging.info(f"Created timestamp column with {valid_count} valid entries")
    if invalid_count > 0:
        logging.warning(f"{invalid_count} rows have invalid timestamps (set to NaT)")
    
    return result_df

def extract_temporal_features(df):
    """Extract date and hour from the timestamp column."""
    result_df = df.copy()
    
    # Check for timestamp column
    if 'timestamp' not in result_df.columns:
        logging.error("No 'timestamp' column found")
        raise ValueError("No 'timestamp' column found")
    
    # Ensure timestamp is datetime type
    if not pd.api.types.is_datetime64_any_dtype(result_df['timestamp']):
        logging.info("Converting 'timestamp' to datetime64")
        result_df['timestamp'] = pd.to_datetime(result_df['timestamp'], errors='coerce')
    
    # Filter valid timestamps
    valid_rows = result_df['timestamp'].notna()
    if valid_rows.sum() == 0:
        logging.warning("No valid timestamps to extract features from")
        return result_df.assign(date=pd.NaT, hour=np.nan)
    
    # Extract features
    result_df.loc[valid_rows, 'date'] = result_df.loc[valid_rows, 'timestamp'].dt.date
    result_df.loc[valid_rows, 'hour'] = result_df.loc[valid_rows, 'timestamp'].dt.hour
    logging.info(f"Extracted temporal features for {valid_rows.sum()} rows")
    
    return result_df

def group_data(df):
    """Group data by date, hour, PRECINCT, and TYP_DESC, counting calls."""
    result_df = df.copy()
    
    # Rename NYPD_PCT_CD to PRECINCT
    if 'NYPD_PCT_CD' in result_df.columns:
        result_df = result_df.rename(columns={'NYPD_PCT_CD': 'PRECINCT'})
    
    # Check required columns
    required_cols = ['date', 'hour', 'PRECINCT', 'TYP_DESC']
    missing_cols = [col for col in required_cols if col not in result_df.columns]
    if missing_cols:
        logging.error(f"Missing columns for grouping: {missing_cols}")
        raise ValueError(f"Missing columns: {missing_cols}")
    
    # Filter out rows with missing values in required columns
    valid_df = result_df.dropna(subset=required_cols)
    if len(valid_df) == 0:
        logging.warning("No valid data remains after filtering")
        return pd.DataFrame(columns=required_cols + ['call_count'])
    
    # Group and count
    grouped_df = valid_df.groupby(required_cols).size().reset_index(name='call_count')
    logging.info(f"Grouped data into {len(grouped_df)} rows")
    
    return grouped_df

def preprocess_data(df, output_path="data/grouped_data.csv"):
    """Full preprocessing pipeline, saving output to a CSV."""
    logging.info("Starting data preprocessing")
    
    # Step 1: Parse timestamps
    df = parse_timestamps(df)
    
    # Step 2: Extract temporal features
    df = extract_temporal_features(df)
    
    # Step 3: Group data
    grouped_df = group_data(df)
    
    # Save the preprocessed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    grouped_df.to_csv(output_path, index=False)
    logging.info(f"Saved preprocessed data to {output_path}")
    
    logging.info("Data preprocessing completed")
    return grouped_df

if __name__ == "__main__":
    # Load your dataset (adjust path as needed)
    input_path = "data/nypd_calls.csv"
    df = pd.read_csv(input_path)
    logging.info(f"Successfully loaded CSV from {input_path}")
    
    # Preprocess the data and save
    processed_df = preprocess_data(df)
    print(processed_df.head())