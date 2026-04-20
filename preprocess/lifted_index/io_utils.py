import xarray as xr
import logging
from pathlib import Path

def save_li_netcdf(out_file_path, ds_template, all_times, all_li, cmip_version, description):
    """
    Universally reconstructs and saves the Lifted Index NetCDF using a template.
    Works identically for CMIP5, CMIP6 CMORized, and CMIP6 Native.
    """
    logging.info("Reconstructing NetCDF using Spatial Template...")
    
    # Build fresh dataset bound strictly to the new summer-only hourly time axis
    ds_out = xr.Dataset(
        data_vars={"LI": (["time", "rlat", "rlon"], all_li)},
        coords={
            "time": all_times,
            "rlat": ds_template.rlat.values, 
            "rlon": ds_template.rlon.values
        }
    )
    
    # Safely assign 2D spatial coordinates 
    if 'lat' in ds_template.coords or 'lat' in ds_template.variables:
        ds_out = ds_out.assign_coords({
            "lat": ds_template.lat,
            "lon": ds_template.lon
        })
        
    # Manually copy grid mapping projection
    if 'rotated_pole' in ds_template.variables:
        ds_out['rotated_pole'] = ds_template['rotated_pole'].copy()
        
    # Assign metadata attributes
    ds_out['time'].attrs = {'standard_name': 'time', 'long_name': 'time'}
    ds_out['rlat'].attrs = ds_template['rlat'].attrs
    ds_out['rlon'].attrs = ds_template['rlon'].attrs
    ds_out.attrs = ds_template.attrs 

    ds_out["LI"].attrs = {
        "units": "K", 
        "long_name": "Most Unstable LI (Bolton, Strict)",
        "description": description
    }
    
    if "rotated_pole" in ds_template.variables:
        ds_out["LI"].attrs["grid_mapping"] = "rotated_pole"

    # Ensure output directory exists and save
    out_path = Path(out_file_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    ds_out.to_netcdf(out_path, engine='netcdf4')
    logging.info(f"Successfully saved to: {out_path.resolve()}")