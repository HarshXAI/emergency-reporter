import os
import requests
from pathlib import Path
import pandas as pd
import geopandas as gpd

def download_file(url, local_path):
    """
    Download a file from a URL and save it to a local path.
    Creates the directory for the local path if it doesn't exist.
    
    Args:
        url (str): URL of the file to download
        local_path (str): Local path where the file should be saved
    
    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(local_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        
        # Download the file
        print(f"Downloading from {url}...")
        response = requests.get(url, stream=True)
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Save the file
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Successfully downloaded file to {local_path}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Network error occurred: {e}")
        return False
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error occurred: {e}")
        return False
    except requests.exceptions.Timeout as e:
        print(f"Timeout error occurred: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

def load_csv(file_path):
    """
    Load a CSV file into a pandas DataFrame.
    
    Args:
        file_path (str): Path to the CSV file
    
    Returns:
        pandas.DataFrame: DataFrame containing the CSV data
        
    Raises:
        FileNotFoundError: If the specified file does not exist
        pd.errors.EmptyDataError: If the file is empty
        pd.errors.ParserError: If the file cannot be parsed as CSV
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at '{file_path}' does not exist.")
    
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded CSV from {file_path}")
        return df
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"The file at '{file_path}' is empty.")
    except pd.errors.ParserError:
        raise pd.errors.ParserError(f"The file at '{file_path}' could not be parsed as CSV.")
    except Exception as e:
        raise Exception(f"An error occurred while loading the CSV: {str(e)}")

def load_geojson(file_path):
    """
    Load a GeoJSON file into a GeoPandas GeoDataFrame.
    
    Args:
        file_path (str): Path to the GeoJSON file
    
    Returns:
        geopandas.GeoDataFrame: GeoDataFrame containing the GeoJSON data
        
    Raises:
        FileNotFoundError: If the specified file does not exist
        ValueError: If the file cannot be parsed as GeoJSON
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at '{file_path}' does not exist.")
    
    try:
        gdf = gpd.read_file(file_path, driver='GeoJSON')
        print(f"Successfully loaded GeoJSON from {file_path}")
        return gdf
    except ValueError:
        raise ValueError(f"The file at '{file_path}' could not be parsed as GeoJSON.")
    except Exception as e:
        raise Exception(f"An error occurred while loading the GeoJSON: {str(e)}")

if __name__ == "__main__":
    # Example usage
    download_file("https://example.com/file.csv", "data/downloaded/file.csv")
    
    # Example usage of load_csv (commented out to avoid errors when file doesn't exist)
    # df = load_csv("data/downloaded/file.csv")
    # print(df.head())
    
    # Example usage of load_geojson (commented out to avoid errors when file doesn't exist)
    # gdf = load_geojson("data/downloaded/boundaries.geojson")
    # print(gdf.head())
