import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import xarray as xr
import numpy as np
import argparse
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc

from calc_lifted_index import get_most_unstable_li
from io_utils import save_li_netcdf

xr.set_options(file_cache_maxsize=1)

def get_native_filepath(base_dir, exp_id, var_name, year):
    """Resolves native CMIP6 CORDEX paths (e.g. /reloclim/het/ICS370L02/T500p/...)"""
    search_path = Path(base_dir) / exp_id / var_name
    # Fixed the .ncz typo and upgraded to rglob for robustness
    pattern = f"{var_name}_*{year}*.ncz"
    files = list(search_path.rglob(pattern))
    
    if not files:
        raise FileNotFoundError(f"Missing file: {search_path}/{pattern}")
    return files[0]

def process_single_month(month, year, base_dir, exp_id, tolerance_hpa, use_lut):
    try:
        ps_path = get_native_filepath(base_dir, exp_id, "PS", year)
        
        # File path uses T500p
        t500_path = get_native_filepath(base_dir, exp_id, "T500p", year)
        
        with xr.open_dataset(t500_path) as ds_t:
            time_array = ds_t.time.values
            in_month = (ds_t.time.dt.month == month).values
            if not np.any(in_month): return None
                
            indices = np.where(in_month)[0]
            start_idx = max(0, indices[0] - 4)  
            end_idx = min(len(time_array) - 1, indices[-1] + 4) 
            time_slice = slice(time_array[start_idx], time_array[end_idx])
            
            # Internal variable is 'T'
            t500_buf = ds_t['T'].sel(time=time_slice).astype(np.float32).load()
            t500_buf_h = t500_buf.resample(time="1H").interpolate("linear")
            master_da = t500_buf_h.sel(time=t500_buf_h.time.dt.month == month)
            master_time = master_da.time
            t_env_500 = master_da.values

        def get_aligned(path, var_name):
            with xr.open_dataset(path) as ds:
                buf = ds[var_name].sel(time=time_slice).astype(np.float32).load()
                buf_h = buf.resample(time="1H").interpolate("linear")
                return buf_h.interp(time=master_time, method="linear").values

        # File path uses QV500p, internal variable is 'QV'
        q500_path = get_native_filepath(base_dir, exp_id, "QV500p", year)
        q_env_500 = get_aligned(q500_path, "QV")
        
        # PS usually shares the name internally and externally
        ps_vals = get_aligned(ps_path, "PS")

        src_levels_data = {}
        for p_src in [925, 850, 700]:
            # File paths use the 'p' suffix
            t_path = get_native_filepath(base_dir, exp_id, f"T{p_src}p", year)
            q_path = get_native_filepath(base_dir, exp_id, f"QV{p_src}p", year)
            
            # Internal variables are strictly 'T' and 'QV'
            t_src = get_aligned(t_path, "T")
            q_src = get_aligned(q_path, "QV")
            
            src_levels_data[p_src] = (t_src, q_src)

        li_final_arr = get_most_unstable_li(
            t_env_500, q_env_500, ps_vals, src_levels_data, tolerance_hpa, use_lut
        )
            
        del t500_buf_h, master_da, q_env_500, ps_vals, src_levels_data
        gc.collect()
        return (month, master_time.values, li_final_arr)

    except Exception as e:
        import traceback
        return (month, None, traceback.format_exc())

def process_year(year, args):
    output_dir = Path(args.output_basedir) / "cmip6" / args.output_model_id
    out_file = output_dir / f"LI_{args.output_model_id}_{year}.nc"
    if out_file.exists(): return

    results = []
    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_single_month, m, year, args.input_base_dir, args.exp_id, args.tolerance_hpa, args.use_lut) for m in args.months]
        for future in as_completed(futures):
            res = future.result()
            if res and res[1] is not None:
                results.append(res)
            elif res:
                raise RuntimeError(f"Worker failed on month {res[0]}:\n{res[2]}")

    if not results: return
    results.sort(key=lambda x: x[0])
    all_times = np.concatenate([r[1] for r in results])
    all_li = np.concatenate([r[2] for r in results], axis=0)

    ps_path = get_native_filepath(args.input_base_dir, args.exp_id, "PS", year)
    with xr.open_dataset(ps_path) as ds_template:
        desc = "Calculated from EURO-CORDEX CMIP6 (Non-CMORized Native)."
        save_li_netcdf(out_file, ds_template, all_times, all_li, "cmip6_native", desc)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_base_dir", type=str, default="/reloclim/het/")
    parser.add_argument("--exp_id", type=str, required=True, help="e.g., ICS370L02")
    parser.add_argument("--output_basedir", type=str, default="/reloclim/dkn/euro-cordex/data/lifted_index/")
    parser.add_argument("--output_model_id", type=str, required=True)
    parser.add_argument("--start_year", type=int, default=1970)
    parser.add_argument("--end_year", type=int, default=2005)
    parser.add_argument("--months", type=int, nargs='+', default=[5, 6, 7, 8, 9])
    parser.add_argument("--tolerance_hpa", type=float, default=0.0)
    parser.add_argument("--use_lut", action="store_true")
    args = parser.parse_args()

    for year in range(args.start_year, args.end_year + 1):
        process_year(year, args)

if __name__ == "__main__":
    main()