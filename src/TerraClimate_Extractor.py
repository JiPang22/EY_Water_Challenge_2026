# Generated and Patched Extractor for Inference

import warnings
warnings.filterwarnings("ignore")

# Data manipulation and analysis
import numpy as np
import pandas as pd

# Multi-dimensional arrays and datasets (e.g., NetCDF, Zarr)
import xarray as xr

from scipy.spatial import cKDTree

# Planetary Computer tools for STAC API access and authentication
import pystac_client
import planetary_computer as pc

from datetime import date
from tqdm import tqdm
import os

def load_terraclimate_dataset():
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )
    collection = catalog.get_collection("terraclimate")
    asset = collection.assets["zarr-abfs"]

    if "xarray:storage_options" in asset.extra_fields:
        ds = xr.open_zarr(
            asset.href,
            storage_options=asset.extra_fields["xarray:storage_options"],
            consolidated=True,
        )
    else:
        ds = xr.open_dataset(
            asset.href,
            **asset.extra_fields["xarray:open_kwargs"],
        )

    return ds

# --- Filtering function (kept identical) ---
def filterg(ds, var):
    ds_2011_2015 = ds[var].sel(time=slice("2011-01-01", "2015-12-31"))

    df_var_append = []
    for i in tqdm(range(len(ds_2011_2015.time))):
        df_var = ds_2011_2015.isel(time=i).to_dataframe().reset_index()
        df_var_filter = df_var[
            (df_var['lat'] > -35.18) & (df_var['lat'] < -21.72) &
            (df_var['lon'] > 14.97) & (df_var['lon'] < 32.79)
        ]
        df_var_append.append(df_var_filter)

    df_var_final = pd.concat(df_var_append, ignore_index=True)
    print(f"Filtering for {var} completed")

    df_var_final['time'] = df_var_final['time'].astype(str)

    # Column mapping
    col_mapping = {"lat": "Latitude", "lon": "Longitude", "time": "Sample Date"}
    df_var_final = df_var_final.rename(columns=col_mapping)

    return df_var_final


# --- Climate variable assignment function (unchanged logic) ---
def assign_nearest_climate(sa_df, climate_df, var_name):
    """
    Map nearest climate variable values to a new DataFrame 
    containing only the specified variable column.
    """
    sa_coords = np.radians(sa_df[['Latitude', 'Longitude']].values)
    climate_coords = np.radians(climate_df[['Latitude', 'Longitude']].values)

    tree = cKDTree(climate_coords)
    dist, idx = tree.query(sa_coords, k=1)

    nearest_points = climate_df.iloc[idx].reset_index(drop=True)

    sa_df = sa_df.reset_index(drop=True)
    sa_df[['nearest_lat', 'nearest_lon']] = nearest_points[['Latitude', 'Longitude']]

    sa_df['Sample Date'] = pd.to_datetime(sa_df['Sample Date'], dayfirst=True, errors='coerce')
    climate_df['Sample Date'] = pd.to_datetime(climate_df['Sample Date'], dayfirst=True, errors='coerce')

    climate_values = []

    for i in tqdm(range(len(sa_df)), desc=f"Mapping {var_name.upper()} values"):
        sample_date = sa_df.loc[i, 'Sample Date']
        nearest_lat = sa_df.loc[i, 'nearest_lat']
        nearest_lon = sa_df.loc[i, 'nearest_lon']

        subset = climate_df[
            (climate_df['Latitude'] == nearest_lat) &
            (climate_df['Longitude'] == nearest_lon)
        ]

        if subset.empty:
            climate_values.append(np.nan)
            continue

        nearest_idx = (subset['Sample Date'] - sample_date).abs().idxmin()
        climate_values.append(subset.loc[nearest_idx, var_name])

    output_df = pd.DataFrame({var_name: climate_values})

    
    return output_df

Water_Quality_df=pd.read_csv('water_quality_training_dataset.csv')
Water_Quality_df.head()

Water_Quality_df.shape

# Load TerraClimate dataset, filter (time,region,parameter), filter for nearest parameter values
ds = load_terraclimate_dataset()
tc_parameter = filterg(ds,'pet')
Terraclimate_training_df = assign_nearest_climate(Water_Quality_df, tc_parameter, 'pet')

Terraclimate_training_df['Latitude'] = Water_Quality_df['Latitude']
Terraclimate_training_df['Longitude'] = Water_Quality_df['Longitude']
Terraclimate_training_df['Sample Date'] = Water_Quality_df['Sample Date']
Terraclimate_training_df = Terraclimate_training_df[['Latitude', 'Longitude', 'Sample Date', 'pet']]
Terraclimate_training_df.to_csv('terraclimate_features_training.csv', index=False)

# Preview File
Terraclimate_training_df.head()

Validation_df=pd.read_csv('submission_template.csv')
Validation_df.head()

Validation_df.shape

# Load TerraClimate dataset, filter (time,region,parameter), filter for nearest parameter values
Terraclimate_validation_df = assign_nearest_climate(Validation_df, tc_parameter, 'pet')

Terraclimate_validation_df['Latitude'] = Validation_df['Latitude']
Terraclimate_validation_df['Longitude'] = Validation_df['Longitude']
Terraclimate_validation_df['Sample Date'] = Validation_df['Sample Date']
Terraclimate_validation_df = Terraclimate_validation_df[['Latitude', 'Longitude', 'Sample Date', 'pet']]
Terraclimate_validation_df.to_csv('terraclimate_features_validation.csv', index=False)

# Preview File
Terraclimate_validation_df.head()
