import numpy as np
from numba import vectorize
import warnings

# --- 1. VECTORIZED C-COMPILED PHYSICS (Bolton 1980) ---

@vectorize(cache=True)
def compute_theta_e(T, p_pa, q):
    """
    Calculates Equivalent Potential Temperature (Theta-E) for a moist air parcel.
    
    Based on the exact formula from Bolton (1980), "The Computation of Equivalent 
    Potential Temperature", Monthly Weather Review. Compiled to C-level machine code 
    via Numba for high-performance processing across large spatio-temporal arrays.
    
    Parameters:
    -----------
    T : float
        Temperature of the parcel [K]
    p_pa : float
        Pressure of the parcel [Pa]
    q : float
        Specific humidity of the parcel [kg/kg]
        
    Returns:
    --------
    float
        Equivalent Potential Temperature (Theta-E) [K]
    """
    e_pa = p_pa * q / (0.622 + 0.378 * q)
    e_hpa = e_pa / 100.0
    
    e_safe = max(e_hpa, 0.001)
    ln_e = np.log(e_safe / 6.112)
    td_c = (243.5 * ln_e) / (17.67 - ln_e)
    Td = td_c + 273.15
    
    Td_safe = max(Td, 57.0) 
    denom = (1.0 / (Td_safe - 56.0)) + (np.log(T / Td_safe) / 800.0)
    tlcl = (1.0 / denom) + 56.0
    
    theta = T * (100000.0 / p_pa) ** 0.2854
    r = q / (1.0 - q)
    
    exp_arg = (3376.0 / tlcl - 2.54) * r * (1.0 + 0.81 * r)
    exp_arg = max(-30.0, min(exp_arg, 30.0))
    return theta * np.exp(exp_arg)

@vectorize(cache=True)
def solve_t500_exact(te_target):
    """
    Iterative Newton-Raphson solver to find the Temperature of a lifted parcel at 500 hPa.
    
    Reverses the Theta-E equation to solve for the temperature a parcel would have 
    if lifted pseudo-adiabatically to 500 hPa, given its initial Theta-E.
    
    Parameters:
    -----------
    te_target : float
        The initial Equivalent Potential Temperature (Theta-E) of the lifted parcel [K]
        
    Returns:
    --------
    float
        Temperature of the parcel at 500 hPa [K]
    """
    if not np.isfinite(te_target) or te_target <= 200.0 or te_target >= 500.0:
        return np.nan
    
    p_target = 50000.0
    T = 250.0
    
    for _ in range(8):
        T = max(150.0, min(T, 350.0))
        T_c = T - 273.15
        es = 611.2 * np.exp(17.67 * T_c / (T_c + 243.5))
        qs = 0.622 * es / (p_target - 0.378 * es)
        r = qs / (1.0 - qs)
        theta = T * (100000.0 / p_target) ** 0.2854
        
        exp_arg = (3376.0 / T - 2.54) * r * (1.0 + 0.81 * r)
        exp_arg = max(-30.0, min(exp_arg, 30.0))
        te_calc = theta * np.exp(exp_arg)
        
        f_val = te_calc - te_target
        
        dt = 0.1
        T_next = T + dt
        T_c_n = T_next - 273.15
        es_n = 611.2 * np.exp(17.67 * T_c_n / (T_c_n + 243.5))
        qs_n = 0.622 * es_n / (p_target - 0.378 * es_n)
        r_n = qs_n / (1.0 - qs_n)
        th_n = T_next * (100000.0 / p_target) ** 0.2854
        
        exp_n = (3376.0 / T_next - 2.54) * r_n * (1.0 + 0.81 * r_n)
        exp_n = max(-30.0, min(exp_n, 30.0))
        te_n_calc = th_n * np.exp(exp_n)
        
        df_dt = (te_n_calc - te_calc) / dt
        step = f_val / (df_dt + 1e-6)
        step = max(-10.0, min(step, 10.0))
        T = T - step
        
    return T

# Lookup Table (LUT) Initialization for faster processing if exact solver is bypassed
LUT_THETA_E = np.arange(200.0, 500.0, 0.001, dtype=np.float32)
LUT_T500 = solve_t500_exact(LUT_THETA_E)

def get_most_unstable_li(t_env_500, q_env_500, ps_vals, src_levels_data, tolerance_hpa=0.0, use_lut=False):
    """
    Computes the most unstable Lifted Index (LI) from multiple source pressure levels.
    
    Calculates the LI for parcels lifted from various lower tropospheric levels 
    (e.g., 925, 850, 700 hPa) to 500 hPa, masks out subterranean parcels based on 
    surface pressure, and returns the minimum (most unstable) LI value across all levels.
    
    Parameters:
    -----------
    t_env_500 : ndarray
        Environmental temperature at 500 hPa [K]
    q_env_500 : ndarray
        Environmental specific humidity at 500 hPa [kg/kg]
    ps_vals : ndarray
        Surface pressure to identify and mask subterranean source levels [Pa]
    src_levels_data : dict
        Dictionary formatted as { pressure_level_hpa: (Temperature_array, Specific_Humidity_array) }.
        Example: { 850: (t850_array, q850_array) }
    tolerance_hpa : float, optional
        Tolerance margin for subterranean masking [hPa] (default is 0.0)
    use_lut : bool, optional
        If True, uses a pre-computed lookup table for lifting parcels to 500 hPa 
        instead of the exact Newton-Raphson solver (default is False)
        
    Returns:
    --------
    ndarray
        Array containing the Most Unstable Lifted Index [K], masked with NaNs where 
        all parcels are subterranean.
    """
    li_candidates = []
    
    for p_src, (t_src, q_src) in src_levels_data.items():
        p_src_pa = np.float32(p_src * 100.0)
        
        # 1. Parcel Theta-E
        te_src = compute_theta_e(t_src, p_src_pa, q_src)
        
        # 2. Lift to 500 hPa
        if use_lut:
            t_parcel_500 = np.interp(te_src, LUT_THETA_E, LUT_T500)
        else:
            t_parcel_500 = solve_t500_exact(te_src)
            
        # 3. Virtual Temperatures
        T_c_par = t_parcel_500 - 273.15
        es_parcel = 611.2 * np.exp(17.67 * T_c_par / (T_c_par + 243.5))
        qs_parcel = 0.622 * es_parcel / (50000.0 - 0.378 * es_parcel)
        tv_parcel = t_parcel_500 * (1.0 + 0.61 * qs_parcel)
        tv_env = t_env_500 * (1.0 + 0.61 * q_env_500)
        
        # 4. Lifted Index
        li_level = tv_env - tv_parcel
        
        # 5. Mask out subterranean parcels
        limit = ps_vals + (tolerance_hpa * 100.0)
        li_level = np.where(p_src_pa <= limit, li_level, np.nan)
        
        li_candidates.append(li_level)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Returns the most unstable (minimum) LI across all valid vertical levels
        li_final_arr = np.nanmin(np.stack(li_candidates, axis=0), axis=0)
        
    return li_final_arr