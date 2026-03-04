import xarray as xr
import numpy as np
import xesmf as xe
import multiprocessing
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import datetime
import os

# --- CONFIGURATION ---
NUM_CORES = 10

# Input/Output Directories
INPUT_DIR = Path("/reloclim/dkn/data/cerra/total_precipitation")
OUTPUT_DIR = Path("/reloclim/dkn/euro-cordex/data/precip/cerra")

# The Target Grid (Master Template)
TARGET_GRID_FILE = Path("/reloclim/dkn/euro-cordex/data/grid_files/euro-cordex_11_target_grid.nc")

# Weights file
WEIGHTS_FILE = Path("./weights/conservative_cerra_to_cordex.nc")

YEARS = range(2018, 2023)
MONTHS = [5, 6, 7, 8, 9]

# --- HELPER: GENERATE 2D BOUNDS FOR CERRA ---
def generate_2d_bounds(centers):
    ny, nx = centers.shape
    b = np.zeros((ny + 1, nx + 1))
    b[1:-1, 1:-1] = (centers[:-1, :-1] + centers[1:, :-1] + 
                     centers[:-1, 1:] + centers[1:, 1:]) / 4.0
    b[0, 1:-1] = b[1, 1:-1] - (b[2, 1:-1] - b[1, 1:-1])
    b[-1, 1:-1] = b[-2, 1:-1] + (b[-2, 1:-1] - b[-3, 1:-1])
    b[:, 0] = b[:, 1] - (b[:, 2] - b[:, 1])
    b[:, -1] = b[:, -2] + (b[:, -2] - b[:, -3])
    return b

def prep_cerra_grid(ds):
    ds_out = ds.copy()
    rename_dict = {}
    if 'latitude' in ds_out: rename_dict['latitude'] = 'lat'
    if 'longitude' in ds_out: rename_dict['longitude'] = 'lon'
    ds_out = ds_out.rename(rename_dict)
    ds_out = ds_out.set_coords(['lat', 'lon'])
    lon_wrapped = ((ds_out['lon'] + 180) % 360) - 180
    ds_out = ds_out.assign_coords(lon=lon_wrapped)
    lat_b = generate_2d_bounds(ds_out['lat'].values)
    lon_b = generate_2d_bounds(ds_out['lon'].values)
    ds_out = ds_out.assign_coords({
        'lat_b': (('y_b', 'x_b'), lat_b),
        'lon_b': (('y_b', 'x_b'), lon_b)
    })
    return ds_out

def process_file_task(task_tuple):
    file_path, target_grid_path, out_dir = task_tuple
    try:
        with xr.open_dataset(file_path) as ds:
            ds_target = xr.open_dataset(target_grid_path)
            
            # Dynamic Filename based on timestamp
            dt = pd.to_datetime(ds.time.values[0])
            out_file = out_dir / f"TOT_PREC_cerra_{dt.strftime('%Y%m%dT%H30')}.nc"
            if out_file.exists(): return

            ds_in = prep_cerra_grid(ds)
            
            # Initialize Regridder (xESMF uses target bounds automatically via cf_xarray)
            regridder = xe.Regridder(
                ds_in, ds_target, "conservative", 
                filename=str(WEIGHTS_FILE), reuse_weights=True
            )

            # Regrid ('tp' is the variable in source CERRA)
            da_remap = regridder(ds_in['tp'], keep_attrs=False)

            # --- Construct Dataset using Master Template ---
            # Replicates the working Bilinear structure
            var_name_out = "TOT_PREC"
            ds_out = xr.Dataset(
                data_vars={
                    var_name_out: (['time', 'rlat', 'rlon'], da_remap.values.reshape(1, 412, 424)),
                    'rotated_pole': ds_target['rotated_pole'],
                    'lat_bnds': ds_target['lat_bnds'],
                    'lon_bnds': ds_target['lon_bnds'],
                },
                coords={
                    'time': ds_in.time,
                    'rlat': ds_target.rlat,
                    'rlon': ds_target.rlon,
                    'lat': (['rlat', 'rlon'], ds_target.lat.values),
                    'lon': (['rlat', 'rlon'], ds_target.lon.values)
                }
            )

            # --- Replicate Attributes and Fix Metadata ---
            for coord in ['rlat', 'rlon', 'lat', 'lon', 'lat_bnds', 'lon_bnds', 'time']:
                if coord in ds_out:
                    ds_out[coord].attrs = ds_target[coord].attrs.copy()

            ds_out[var_name_out].attrs = {
                'standard_name': "precipitation_amount",
                'long_name': "total precip",
                'units': "kg m-2",
                'grid_mapping': "rotated_pole",
                'coordinates': "lat lon",
                'cell_methods': "time: sum"
            }
            
            # Encoding Fix: Avoid 'coordinates' conflict
            target_encoding = ds_target['precipitation'].encoding.copy()
            target_encoding.pop('coordinates', None)
            target_encoding.update({
                '_FillValue': -1.e+20,
                'missing_value': -1.e+20,
                'dtype': 'float32'
            })
            ds_out[var_name_out].encoding = target_encoding

            # Preserve time_bnds if present
            if 'time_bnds' in ds_target:
                ds_out['time_bnds'] = (('time', 'bnds'), np.full((1, 2), np.nan))
                ds_out['time_bnds'].attrs = ds_target['time_bnds'].attrs.copy()

            # --- Global Attribute Cleanup ---
            ds_out.attrs = ds_target.attrs.copy()
            purge_list = [
                'CDI', 'CDO', 'NCO', 'cdo_openmp_thread_number', 'contact',
                'icon-clm_version', 'references', 'comment', 'project_id', 
                'experiment_id', 'realization', 'ConventionsURL', 'institution'
            ]
            for key in purge_list:
                ds_out.attrs.pop(key, None)

            ds_out.attrs.update({
                'title': "CERRA total precipitation remapped to EURO-CORDEX 0.11",
                'institution': "Wegener Center for Climate and Global Change, University of Graz",
                'contact': "David Kneidinger, david.kneidinger@uni-graz.at",
                'source': "CERRA Regional Reanalysis",
                'history': f"Remapped with xESMF conservative on {datetime.datetime.now().isoformat()}",
                'creation_date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Conventions': "CF-1.4"
            })

            ds_out.to_netcdf(out_file)
            
    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")

def main():
    print("--- Checking/Generating Weights (Conservative) ---")
    WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    sample_files = sorted(list(INPUT_DIR.glob("**/*.nc")))
    if not sample_files: return
    
    ds_sample = prep_cerra_grid(xr.open_dataset(sample_files[0]))
    ds_target = xr.open_dataset(TARGET_GRID_FILE)
    
    xe.Regridder(ds_sample, ds_target, "conservative", 
                 filename=str(WEIGHTS_FILE), reuse_weights=WEIGHTS_FILE.exists())
    print("Weights ready.")

    tasks = []
    for year in YEARS:
        for month in MONTHS:
            ydir = INPUT_DIR / str(year) / f"{month:02d}"
            if not ydir.exists(): continue
            out_ydir = OUTPUT_DIR / str(year) / f"{month:02d}"
            out_ydir.mkdir(parents=True, exist_ok=True)
            for f in sorted(list(ydir.glob("*.nc"))):
                tasks.append((f, TARGET_GRID_FILE, out_ydir))

    print(f"--- Processing {len(tasks)} files on {NUM_CORES} cores ---")
    with multiprocessing.Pool(NUM_CORES) as pool:
        list(tqdm(pool.imap_unordered(process_file_task, tasks), total=len(tasks)))
    print("Done.")

if __name__ == "__main__":
    main()