import xarray as xr
import xesmf as xe
import multiprocessing
from pathlib import Path
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import datetime

# --- CONFIGURATION ---
NUM_CORES = 10

INPUT_DIR = Path("/reloclim/dkn/data/cerra/lifted_index/")
OUTPUT_DIR = Path("/reloclim/dkn/euro-cordex/data/lifted_index/cerra")

TARGET_GRID_FILE = Path("/reloclim/dkn/euro-cordex/data/grid_files/euro-cordex_11_target_grid.nc")
WEIGHTS_FILE = Path("./weights/bilinear_cerra_to_cordex.nc")

YEARS = range(2018, 2023) 
MONTHS = [5, 6, 7, 8, 9]

def prep_cerra_grid_bilinear(ds):
    """Renames and prepares coordinates for xESMF bilinear."""
    ds_out = ds.copy()
    rename_dict = {}
    if 'latitude' in ds_out.variables: rename_dict['latitude'] = 'lat'
    if 'longitude' in ds_out.variables: rename_dict['longitude'] = 'lon'
    if rename_dict:
        ds_out = ds_out.rename(rename_dict)
    ds_out = ds_out.set_coords(['lat', 'lon'])
    if ds_out['lon'].max() > 180:
        ds_out = ds_out.assign_coords(lon=(((ds_out['lon'] + 180) % 360) - 180))
    return ds_out

def process_month_task(task_tuple):
    year, month, target_path, reuse_flag = task_tuple
    
    in_dir = INPUT_DIR / str(year) / f"{month:02d}"
    out_dir = OUTPUT_DIR / str(year) / f"{month:02d}"
    
    if not in_dir.exists(): return
    files = sorted(list(in_dir.glob("*.nc")))
    if not files: return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Target Template
    ds_target = xr.open_dataset(target_path)
    
    # Initialize Regridder
    ds_sample = xr.open_dataset(files[0])
    ds_sample = prep_cerra_grid_bilinear(ds_sample)

    regridder = xe.Regridder(
        ds_sample, ds_target, "bilinear", 
        filename=str(WEIGHTS_FILE), reuse_weights=True  
    )

    for f in files:
        try:
            with xr.open_dataset(f) as ds:
                # 1. Generate dynamic filename
                dt = pd.to_datetime(ds.time.values[0])
                new_filename = f"lifted_index_cerra_{dt.strftime('%Y%m%dT%H00')}.nc"
                out_file = out_dir / new_filename
                
                if out_file.exists(): continue
                
                # 2. Prep and Regrid
                ds_in = prep_cerra_grid_bilinear(ds)
                var_name_in = "LI" if "LI" in ds_in.variables else list(ds_in.data_vars)[0]
                da_remap = regridder(ds_in[var_name_in], keep_attrs=False)
                
                # 3. Construct Dataset using Master Template
                # This mirrors the variable order and bonded coordinates of the target
                ds_out = xr.Dataset(
                    data_vars={
                        'LI': (['time', 'rlat', 'rlon'], da_remap.values.reshape(1, 412, 424)),
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

                # 4. Replicate Coordinate Attributes exactly from Target
                for coord in ['rlat', 'rlon', 'lat', 'lon', 'lat_bnds', 'lon_bnds', 'time']:
                    if coord in ds_out:
                        ds_out[coord].attrs = ds_target[coord].attrs.copy()
                
                # 5. Set Main Variable Attributes
                ds_out['LI'].attrs = {
                    'standard_name': "lifted_index",
                    'long_name': "Lifted Index",
                    'units': "K",
                    'grid_mapping': "rotated_pole",
                    'coordinates': "lat lon"
                }
                
                # --- THE FIX ---
                # Copy encoding, but remove 'coordinates' to avoid the conflict
                target_encoding = ds_target['precipitation'].encoding.copy()
                target_encoding.pop('coordinates', None) 
                
                ds_out['LI'].encoding = target_encoding

                # 6. Global Attribute Cleanup
                ds_out.attrs = ds_target.attrs.copy()
                
                # List of attributes to purge (ICON-specific)
                purge_list = [
                    'CDI', 'CDO', 'NCO', 'cdo_openmp_thread_number', 'contact',
                    'icon-clm_version', 'references', 'comment', 'project_id', 
                    'experiment_id', 'realization', 'ConventionsURL', 'institution'
                ]
                for key in purge_list:
                    ds_out.attrs.pop(key, None)

                # Update with current run info
                ds_out.attrs.update({
                    'title': "CERRA lifted index remapped to EURO-CORDEX 0.11",
                    'institution': "Wegener Center for Climate and Global Change, University of Graz",
                    'contact': "David Kneidinger, david.kneidinger@uni-graz.at",
                    'source': "CERRA Regional Reanalysis",
                    'history': f"Remapped with xESMF bilinear on {datetime.datetime.now().isoformat()}",
                    'creation_date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'Conventions': "CF-1.4"
                })

                ds_out.to_netcdf(out_file)
                
        except Exception as e:
            print(f"Error processing {f.name}: {e}")

def main():
    print("--- Checking Weights File (Bilinear) ---")
    WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    ds_target = xr.open_dataset(TARGET_GRID_FILE)
    
    # Locate sample for weight check
    sample_file = next(INPUT_DIR.glob("**/*.nc"), None)
    if not sample_file:
        print("No CERRA files found.")
        return

    ds_in = prep_cerra_grid_bilinear(xr.open_dataset(sample_file))
    
    xe.Regridder(
        ds_in, ds_target, "bilinear", 
        filename=str(WEIGHTS_FILE), reuse_weights=WEIGHTS_FILE.exists()
    )
    print("Weights ready.")

    tasks = [(y, m, TARGET_GRID_FILE, True) for y in YEARS for m in MONTHS]
    
    print(f"--- Processing {len(tasks)} months on {NUM_CORES} cores ---")
    with multiprocessing.Pool(NUM_CORES) as pool:
        list(tqdm(pool.imap_unordered(process_month_task, tasks), total=len(tasks)))

if __name__ == "__main__":
    main()