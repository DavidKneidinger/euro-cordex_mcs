import subprocess
import sys
import logging
from datetime import datetime
from pathlib import Path
# nohup ID: 1972603
# --- CONFIGURATION ---
START_YEAR = 1970
END_YEAR = 2005
USE_LUT = True

OUTPUT_BASE_DIR = "/reloclim/dkn/euro-cordex/data/lifted_index/"

# 1. CMIP5 Target Runs (Uses cmip5_li.py)
CMIP5_MODELS = None
#CMIP5_MODELS = [
#    "COSMO-crCLIM-v1-1.CNRM-CERFACS-CNRM-CM5.historical",
#    "COSMO-crCLIM-v1-1.MPI-M-MPI-ESM-LR.historical",
#]

# 2. CMIP6 CMORized Runs (Uses cmip6_cmorized_li.py)
CMIP6_CMOR_MODELS = None
#CMIP6_CMOR_MODELS = {
#    "ICON-CLM-202407-1-1.MPI-ESM1-2-HR.historical": (
#        "/reloclim/het/CLMcom-DWD/MPI-ESM1-2-HR/historical/r1i1p1f1/ICON-CLM-202407-1-1/v1-r1/1hr",
#        "/reloclim/het/CLMcom-DWD/MPI-ESM1-2-HR/historical/r1i1p1f1/ICON-CLM-202407-1-1/v1-r1/6hr"
#    ),
#    "ICON-CLM-202407-1-1.MIROC6.historical": (
#        "/reloclim/het/CLMcom-KIT/MIROC6/historical/r1i1p1f1/ICON-CLM-202407-1-1/v1-r1/1hr",
#        "/reloclim/het/CLMcom-KIT/MIROC6/historical/r1i1p1f1/ICON-CLM-202407-1-1/v1-r1/6hr"
#    )
#}

# 3. CMIP6 Non-CMORized Runs (Uses cmip6_non_cmorized_li.py)
CMIP6_NATIVE_MODELS = {
    "IEHISTL01": "ICON-CLM-202407-1-1.EC-Earth3-Veg.historical",
    "ICHISTL02": "ICON-CLM-202407-1-1.CMCC-CM2-SR5.historical",
    "IFHISTL01": "ICON-CLM-202407-1-1.CNRM-ESM2-1.historical",
    #"IAEVALL02": "ICON-CLM-202407-1-1.ERA5-v1-r1.evaluation",
}

# --- OUTPUT CHECKER ---

def is_model_completed(group_folder, model_id, start_year, end_year):
    """
    Checks if all expected output files for a given model and timeframe already exist.
    Returns True if fully completed, False if any year is missing.
    """
    target_dir = Path(OUTPUT_BASE_DIR) / group_folder / model_id
    
    # If the folder doesn't exist yet, we definitely need to run it
    if not target_dir.exists():
        return False
        
    for year in range(start_year, end_year + 1):
        expected_file = target_dir / f"LI_{model_id}_{year}.nc"
        if not expected_file.exists():
            return False  # Found a missing year, trigger the run
            
    return True # All files exist!


# --- EXECUTION ---

def run_script(command, model_name):
    """Executes a subprocess command safely."""
    try:
        subprocess.run(command, check=True)
        logging.info(f"SUCCESS: Completed {model_name}")
    except subprocess.CalledProcessError as e:
        logging.error(f"ERROR: Processing failed for {model_name} (Exit code: {e.returncode}).")
    except FileNotFoundError:
        logging.error(f"CRITICAL ERROR: Could not find script {command[1]}.")
        sys.exit(1)

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - MASTER - %(levelname)s - %(message)s')
    start_time = datetime.now()

    # --- RUN CMIP5 ---
    if CMIP5_MODELS:
        logging.info(f"--- Starting CMIP5 Processing ({len(CMIP5_MODELS)} models) ---")
        for model in CMIP5_MODELS:
            if is_model_completed("cmip5", model, START_YEAR, END_YEAR):
                logging.info(f"SKIPPING {model}: All output files ({START_YEAR}-{END_YEAR}) already exist.")
                continue
                
            cmd = [sys.executable, "cmip5_li.py", "--model_ver", model, 
                   "--start_year", str(START_YEAR), "--end_year", str(END_YEAR)]
            if USE_LUT: cmd.append("--use_lut")
            run_script(cmd, model)

    # --- RUN CMIP6 CMORIZED ---
    if CMIP6_CMOR_MODELS:
        logging.info(f"--- Starting CMIP6 CMORized Processing ({len(CMIP6_CMOR_MODELS)} models) ---")
        for model_id, dirs in CMIP6_CMOR_MODELS.items():
            if is_model_completed("cmip6", model_id, START_YEAR, END_YEAR):
                logging.info(f"SKIPPING {model_id}: All output files ({START_YEAR}-{END_YEAR}) already exist.")
                continue
                
            cmd = [sys.executable, "cmip6_cmorized_li.py", 
                   "--output_model_id", model_id,
                   "--input_dir_1hr", dirs[0],
                   "--input_dir_6hr", dirs[1],
                   "--start_year", str(START_YEAR), "--end_year", str(END_YEAR)]
            if USE_LUT: cmd.append("--use_lut")
            run_script(cmd, model_id)

    # --- RUN CMIP6 NON-CMORIZED ---
    if CMIP6_NATIVE_MODELS:
        logging.info(f"--- Starting CMIP6 Native Processing ({len(CMIP6_NATIVE_MODELS)} models) ---")
        for exp_id, model_id in CMIP6_NATIVE_MODELS.items():
            if is_model_completed("cmip6", model_id, START_YEAR, END_YEAR):
                logging.info(f"SKIPPING {model_id}: All output files ({START_YEAR}-{END_YEAR}) already exist.")
                continue
                
            cmd = [sys.executable, "cmip6_non_cmorized_li.py",
                   "--exp_id", exp_id,
                   "--output_model_id", model_id,
                   "--start_year", str(START_YEAR), "--end_year", str(END_YEAR)]
            if USE_LUT: cmd.append("--use_lut")
            run_script(cmd, model_id)

    duration = datetime.now() - start_time
    logging.info("=" * 60)
    logging.info(f"All done! Total pipeline execution time: {duration}")
    logging.info("=" * 60)

if __name__ == "__main__":
    main()