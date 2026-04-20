import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import xarray as xr
import numpy as np
import argparse
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc

# Import our custom modules
from calc_lifted_index import get_most_unstable_li
from io_utils import save_li_netcdf

xr.set_options(file_cache_maxsize=1)

LEVELS = [500, 700, 850, 925]

def get_cmor_filepath(base_dir, var_name, year):
    """
    Dynamically resolves the CMORized file path using wildcards to handle 
    the complex timestamp strings (e.g., 201401010000-201412312300).
    Uses rglob to recursively search inside hidden version folders (e.g., v20240920/).
    """
    search_path = Path(base_dir) / var_name
    # Match the variable name prefix and the specific year in the timestamp
    pattern = f"{var_name}_*_{year}*.nc"
    
    # CHANGED: use rglob() instead of glob() to search subdirectories
    files = list(search_path.rglob(pattern))
    
    if not files:
        raise FileNotFoundError(f"Missing file: {search_path}/**/{pattern}")
    
    # If multiple versions exist, usually you want the latest one, 
    # but returning the first one found is usually safe if there's only one version folder.
    return files[0]

def process_single_month(month, year, dir_1hr, dir_6hr, tolerance_hpa, use_lut):
    try:
        ps_path = get_cmor_filepath(dir_1hr, "ps", year)
        
        # Load ta500 as the master time axis template
        t500_path = get_cmor_filepath(dir_6hr, "ta500", year)
        
        with xr.open_dataset(t500_path) as ds_t:
            time_array = ds_t.time.values
            in_month = (ds_t.time.dt.month == month).values
            
            if not np.any(in_month):
                return None
                
            indices = np.where(in_month)[0]
            start_idx = max(0, indices[0] - 4)  
            end_idx = min(len(time_array) - 1, indices[-1] + 4) 
            time_slice = slice(time_array[start_idx], time_array[end_idx])
            
            # Master Time Axis (Interpolated to 1H)
            t500_buf = ds_t['ta500'].sel(time=time_slice).astype(np.float32).load()
            t500_buf_h = t500_buf.resample(time="1H").interpolate("linear")
            master_da = t500_buf_h.sel(time=t500_buf_h.time.dt.month == month)
            master_time = master_da.time
            t_env_500 = master_da.values

        def get_aligned(path, var_name):
            with xr.open_dataset(path) as ds:
                buf = ds[var_name].sel(time=time_slice).astype(np.float32).load()
                # If it's already 1hr (like ps), resampling to 1H does nothing but ensures alignment
                buf_h = buf.resample(time="1H").interpolate("linear")
                return buf_h.interp(time=master_time, method="linear").values

        # Load environment variables
        q500_path = get_cmor_filepath(dir_6hr, "hus500", year)
        q_env_500 = get_aligned(q500_path, "hus500")
        ps_vals = get_aligned(ps_path, "ps")

        # Load source levels into the dictionary format expected by calc_lifted_index
        src_levels_data = {}
        for p_src in [925, 850, 700]:
            t_path = get_cmor_filepath(dir_6hr, f"ta{p_src}", year)
            q_path = get_cmor_filepath(dir_6hr, f"hus{p_src}", year)
            
            t_src = get_aligned(t_path, f"ta{p_src}")
            q_src = get_aligned(q_path, f"hus{p_src}")
            
            src_levels_data[p_src] = (t_src, q_src)

        # Execute the pure math core
        li_final_arr = get_most_unstable_li(
            t_env_500=t_env_500, 
            q_env_500=q_env_500, 
            ps_vals=ps_vals, 
            src_levels_data=src_levels_data, 
            tolerance_hpa=tolerance_hpa, 
            use_lut=use_lut
        )
            
        del t500_buf_h, master_da, q_env_500, ps_vals, src_levels_data
        gc.collect()
            
        return (month, master_time.values, li_final_arr)

    except Exception as e:
        import traceback
        return (month, None, traceback.format_exc())


def process_year(year, args):
    # Construct output path: /base_dir/cmip6/MODEL_ID/LI_MODEL_ID_YEAR.nc
    output_dir = Path(args.output_basedir) / "cmip6" / args.output_model_id
    out_file = output_dir / f"LI_{args.output_model_id}_{year}.nc"
    
    if out_file.exists():
        logging.info(f"File already exists: {out_file.name}. Skipping.")
        return

    results = []
    
    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = []
        for month in args.months:
            futures.append(
                executor.submit(process_single_month, month, year, 
                                args.input_dir_1hr, args.input_dir_6hr, 
                                args.tolerance_hpa, args.use_lut)
            )
            
        for future in as_completed(futures):
            res = future.result()
            if res is None: continue 
            
            month, times, data = res
            if times is None:
                logging.error(f"Failed Month {month}: \n{data}")
                raise RuntimeError(f"Multiprocessing worker failed on month {month}.")
                
            results.append((month, times, data))
            logging.info(f"Finished Month: {month} for {year}")

    if not results:
        return

    results.sort(key=lambda x: x[0])
    all_times = np.concatenate([r[1] for r in results])
    all_li = np.concatenate([r[2] for r in results], axis=0)

    # Reconstruct and Save
    ps_path = get_cmor_filepath(args.input_dir_1hr, "ps", year)
    with xr.open_dataset(ps_path) as ds_template:
        desc = "Calculated from EURO-CORDEX CMIP6 (CMORized)."
        save_li_netcdf(out_file, ds_template, all_times, all_li, "cmip6", desc)


def main():
    parser = argparse.ArgumentParser(description="CMIP6 CMORized Lifted Index Calculator")
    
    parser.add_argument("--input_dir_6hr", type=str, required=True, 
                        help="Base dir containing ta500, hus500, etc.")
    parser.add_argument("--input_dir_1hr", type=str, required=True, 
                        help="Base dir containing ps")
    parser.add_argument("--output_basedir", type=str, default="/reloclim/dkn/euro-cordex/data/lifted_index/", 
                        help="Base output directory")
    parser.add_argument("--output_model_id", type=str, required=True, 
                        help="Folder name for output (e.g., ICON-CLM-202407-1-1.MPI-ESM1-2-HR.historical)")
    
    # Added parameters for year and month ranges
    parser.add_argument("--start_year", type=int, default=1970, help="First year to process (default: 1970)")
    parser.add_argument("--end_year", type=int, default=2005, help="Last year to process (default: 2005)")
    parser.add_argument("--months", type=int, nargs='+', default=[5, 6, 7, 8, 9], help="Months to process (default: May-Sep)")
    
    parser.add_argument("--tolerance_hpa", type=float, default=0.0, help="Subterranean masking tolerance (hPa)")
    parser.add_argument("--use_lut", action="store_true")
    
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info(f"Processing {args.output_model_id} from {args.start_year} to {args.end_year}")
    
    for year in range(args.start_year, args.end_year + 1):
        process_year(year, args)

if __name__ == "__main__":
    main()