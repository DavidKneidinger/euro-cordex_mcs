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

def get_cordex_filepath(base_dir, model_ver, var, level, year, freq):
    sub_dir = f"{model_ver}.{freq}"
    search_path = Path(base_dir) / sub_dir
    prefix = f"{var}{level}" if level else f"{var}"
    pattern = f"{prefix}_*_{year}*.nc"
    files = list(search_path.glob(pattern))
    if not files: raise FileNotFoundError(f"Missing: {search_path}/{pattern}")
    return files[0]

def process_single_month(month, year, model_ver, in_3hr, in_6hr, tol, use_lut):
    try:
        ps_path = get_cordex_filepath(in_3hr, model_ver, "ps", None, year, "3hr")
        t500_path = get_cordex_filepath(in_6hr, model_ver, "ta", 500, year, "6hr")
        
        with xr.open_dataset(t500_path) as ds_t:
            time_array = ds_t.time.values
            in_month = (ds_t.time.dt.month == month).values
            if not np.any(in_month): return None
                
            indices = np.where(in_month)[0]
            start_idx = max(0, indices[0] - 4)
            end_idx = min(len(time_array) - 1, indices[-1] + 4)
            time_slice = slice(time_array[start_idx], time_array[end_idx])
            
            t500_buf = ds_t['ta500'].sel(time=time_slice).astype(np.float32).load()
            t500_buf_h = t500_buf.resample(time="1H").interpolate("linear")
            master_da = t500_buf_h.sel(time=t500_buf_h.time.dt.month == month)
            master_time = master_da.time
            t_env_500 = master_da.values

        def get_aligned(path, var_name):
            with xr.open_dataset(path) as ds:
                buf = ds[var_name].sel(time=time_slice).astype(np.float32).load()
                buf_h = buf.resample(time="1H").interpolate("linear")
                return buf_h.interp(time=master_time, method="linear").values

        q500_path = get_cordex_filepath(in_6hr, model_ver, "hus", 500, year, "6hr")
        q_env_500 = get_aligned(q500_path, "hus500")
        ps_vals = get_aligned(ps_path, "ps")

        src_levels_data = {}
        for p_src in [925, 850, 700]: # CMIP5 strict levels
            t_path = get_cordex_filepath(in_6hr, model_ver, "ta", p_src, year, "6hr")
            q_path = get_cordex_filepath(in_6hr, model_ver, "hus", p_src, year, "6hr")
            src_levels_data[p_src] = (get_aligned(t_path, f"ta{p_src}"), get_aligned(q_path, f"hus{p_src}"))

        li_final_arr = get_most_unstable_li(t_env_500, q_env_500, ps_vals, src_levels_data, tol, use_lut)
        del t500_buf_h, master_da, q_env_500, ps_vals, src_levels_data
        gc.collect()
        return (month, master_time.values, li_final_arr)

    except Exception as e:
        import traceback
        return (month, None, traceback.format_exc())

def process_year(year, args):
    output_dir = Path(args.output_basedir) / "cmip5" / args.model_ver
    out_file = output_dir / f"LI_{args.model_ver}_{year}.nc"
    if out_file.exists(): return

    results = []
    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_single_month, m, year, args.model_ver, args.input_dir_3hr, args.input_dir_6hr, args.tolerance_hpa, args.use_lut) for m in args.months]
        for future in as_completed(futures):
            res = future.result()
            if res and res[1] is not None: results.append(res)
            elif res: raise RuntimeError(f"Worker failed on month {res[0]}:\n{res[2]}")

    if not results: return
    results.sort(key=lambda x: x[0])
    all_times = np.concatenate([r[1] for r in results])
    all_li = np.concatenate([r[2] for r in results], axis=0)

    ps_path = get_cordex_filepath(args.input_dir_3hr, args.model_ver, "ps", None, year, "3hr")
    with xr.open_dataset(ps_path) as ds_template:
        save_li_netcdf(out_file, ds_template, all_times, all_li, "cmip5", "Calculated from EURO-CORDEX CMIP5.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir_6hr", type=str, default="/reloclim/ars/INTERACT/CORDEX/")
    parser.add_argument("--input_dir_3hr", type=str, default="/reloclim/dkn/data/euro-cordex-cmip5")
    parser.add_argument("--output_basedir", type=str, default="/reloclim/dkn/euro-cordex/data/lifted_index/")
    parser.add_argument("--model_ver", type=str, required=True)
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