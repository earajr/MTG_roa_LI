import os
# emulate a smaller system
N_CPUS = 4

os.sched_setaffinity(
    0,
    set(range(N_CPUS))
)

# prevent BLAS/OpenMP oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

N_WORKERS = N_CPUS

from netCDF4 import Dataset
import time
from datetime import datetime, timedelta
import numpy as np
from pysteps import io, motion, nowcasts, rcparams, verification
from pysteps.nowcasts.lagrangian_probability import forecast
from pysteps.verification.detcatscores import det_cat_fct
import shutil
import sys
import xarray as xr
import xesmf as xe
from pyproj import Proj
from pysteps.motion.lucaskanade import dense_lucaskanade
from pysteps.motion import constant
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import gaussian_filter, zoom
from scipy.ndimage import distance_transform_edt
from pathlib import Path
import pandas as pd
from satpy import Scene
import matplotlib.pyplot as plt
import hashlib
import json

def read_domains(domain_file):
 
    with open(domain_file, "r") as f:
        domains = json.load(f)["domains"]
 
    return domains

def get_domain_hash(domain_name, domain_info, dx):
 
    hash_dict = {
        "name": domain_name,
        "dx": dx,
        **domain_info
    }
 
    domain_str = json.dumps(
        hash_dict,
        sort_keys=True
    )
 
    return hashlib.md5(
        domain_str.encode("utf-8")
    ).hexdigest()[:10]

def get_grid_hash(src_grid, dst_grid):
    """Create a unique hash from source and destination grid coordinates."""
 
    data = np.concatenate([
        np.asarray(src_grid["lat"]).ravel()[::100],
        np.asarray(src_grid["lon"]).ravel()[::100],
        np.asarray(dst_grid["lat"]).ravel()[::100],
        np.asarray(dst_grid["lon"]).ravel()[::100],
    ])
 
    return hashlib.md5(data.tobytes()).hexdigest()[:10]

def process_single_MTG_LI_file_with_args(args):
    """
    Worker function to read data from MTG)LI files. Due to frequency of file production (1 every 30 seconds) many files a needed to be read
    for a single accumulation period. Readin the files in parallel allows for some speedup.

    Parameters
    ----------
    fil : str
        Input file path
    var : str
        NetCDF variable to extract
    mask: array
        region of interest mask so that only information that is required is gathered.

    Returns
    -------
    result: array
        data from a single file to be combined with other files being read in parallel.

    """

    import numpy as np

    fil, var, shape = args

    print(f"📂 Reading file: {fil}")

    try:
        scn = Scene(filenames=[fil], reader="li_l2_nc")
        scn.load([var])
        scn_values = scn[var].values

        if np.shape(scn_values) == shape:
            result = np.zeros_like(scn_values)

            if var == "accumulated_flash_area":
                valid_mask = np.isfinite(scn_values) & (scn_values > 0)
                result[valid_mask] = 1
            else:
                valid_mask = np.isfinite(scn_values)
                result[valid_mask] = scn_values[valid_mask]

            return result
        else:
            return np.zeros(shape)

    except Exception as e:
        return np.zeros(shape)


def read_and_accumulate_LI(var, shape, LI_indir, LI_accum_period, latest_time, num_obs):
    from satpy.scene import Scene
    from satpy import find_files_and_readers
    
    end_times = [(latest_time - timedelta(minutes=LI_accum_period * i)) for i in range(1, num_obs + 1)]

    diri = LI_indir

    data = np.zeros([num_obs, shape[0], shape[1]])

    for i, end_time in enumerate(end_times):

        end_time = end_time - timedelta(seconds=1)
        start_time = end_time - timedelta(minutes=LI_accum_period) + timedelta(seconds=30)

        var_map = {
            "accumulated_flash_area": "-AFA--",
            "flash_accumulation": "-AF--",
            "flash_radiance": "-AFR--"
        }

        if var not in var_map:
            raise ValueError(f"Unsupported variable: {var}")
        var_str = var_map[var]

        print(var_str)
        print("start ", start_time)
        print("end ", end_time)

        my_files_temp = find_files_and_readers(base_dir=LI_indir, start_time=start_time, end_time=end_time, reader="li_l2_nc")["li_l2_nc"]
        my_files = [f for f in my_files_temp if var_str in f]
        my_files = sorted(my_files)

        expected_num_files = LI_accum_period * 2
        min_num_files = expected_num_files/2

        if len(my_files) == expected_num_files:
            print("All expected files are present!")
        else:
            if len(my_files) >= min_num_files:
                print(f"Warning: Expected {expected_num_files} files but found {len(my_files)}. Accumulation will continue.")
            else:
                raise RuntimeError(
                    f"Insufficient MTG LI data for observation window {i + 1}: "
                    f"expected at least {min_num_files} files, "
                    f"but found {len(my_files)} "
                    f"({start_time} to {end_time})."
                )


        if len(my_files) >= min_num_files:
  
            with ThreadPoolExecutor(max_workers=4) as executor:
                args_list = [(f, var, np.shape(data[0,:,:])) for f in my_files]
                results = list(executor.map(process_single_MTG_LI_file_with_args, args_list))

                print(np.shape(results))

            if var == "accumulated_flash_area":
                for result in results:
                    valid_mask = np.isfinite(result) & (result > 0)
                    data[i][valid_mask] = 1
                units = "no units"
            else:
                for result in results:
                    valid_mask = np.isfinite(result)
                    data[i][valid_mask] += result[valid_mask]
                units = {
                    "flash_radiance": "mW.m-2.sr-1",
                    "flash_accumulation": "flashes/pixel"
                }.get(var, "unknown")

    return data[::-1,:,:]

def read_RoA(var, shape, input_files):
    """
    Read Rain over Africa precipitation fields.
 
    Parameters
    ----------
    var : str
        Expected to be "precipitation".
    shape : tuple
        Grid shape (ny, nx), e.g. (3712, 3712).
    input_files : list
        List of full file paths ordered oldest -> newest.
 
    Returns
    -------
    rainfall : ndarray
        Array of shape (n_files, ny, nx).
    missing_data : ndarray
        Boolean mask of shape (n_files, ny, nx).
        True indicates missing/invalid data.
    """
 
    if var != "precipitation":
        raise ValueError(f"Unsupported variable: {var}")
 
    n_files = len(input_files)
 
    rainfall = np.full(
        (n_files, shape[0], shape[1]),
        np.nan,
        dtype=np.float32
    )
 
    missing_data = np.ones(
        (n_files, shape[0], shape[1]),
        dtype=bool
    )
 
    for i, filename in enumerate(input_files):
 
        print(f"Reading {filename}")
 
        with Dataset(filename) as nc:
 
            precip_var = nc.variables["precipitation"]
 
            # Convert masked values to NaN
            data = precip_var[:]
 
            if np.ma.isMaskedArray(data):
                precip = data.filled(np.nan)
            else:
                precip = data
 
            precip = np.squeeze(precip).astype(np.float32)
 
            # Handle _FillValue (-9999)
            fill_value = getattr(precip_var, "_FillValue", None)
 
            if fill_value is not None:
                precip[precip == fill_value] = np.nan
 
            # Convert +/- infinity and invalid values to NaN
            precip[~np.isfinite(precip)] = np.nan

            precip[precip > 100] = np.nan
 
            if precip.shape != shape:
                raise ValueError(
                    f"File {filename} has shape {precip.shape}, "
                    f"expected {shape}"
                )
 
            rainfall[i, :, :] = precip
 
            # True where data is missing
            missing_data[i, :, :] = np.isnan(precip)
 
            print(
                f"Valid pixels: {np.sum(~missing_data[i])}, "
                f"Missing pixels: {np.sum(missing_data[i])}"
            )

    return rainfall

def generate_tiles(nx, ny, tile_size, overlap):
    stride = tile_size - overlap
    tiles = []

    for y0 in range(0, ny - tile_size + 1, stride):
        for x0 in range(0, nx - tile_size + 1, stride):

            y1 = y0 + tile_size
            x1 = x0 + tile_size

            tiles.append((y0, y1, x0, x1))

    return tiles

def compute_tile_motion(R, tile):

    y0, y1, x0, x1 = tile

    R_tile = R[:, y0:y1, x0:x1]

    V = dense_lucaskanade(R_tile)

    return tile, V

#####################################################################################
#MAIN
#####################################################################################

print("Reading input vars and setting values")

# Input arguments
LI_indir = Path(os.getcwd()+"/../data/MTG_LI_input")
R_indir = Path(os.getcwd()+"/../data/roa_retrievals")

indate = sys.argv[1]
intime = sys.argv[2]

date = datetime.strptime(indate + intime, "%Y%m%d%H%M")

# Output directory
R_outdir = Path(os.getcwd()+f"/../data/roa_extrapolations/{indate}")

R_outdir.mkdir(parents=True, exist_ok=True)

# Processing parameters
fore_len = 180
timestep = 10

tile_size = 125
overlap   = 40

coarse_factor = 1
alpha = 1.0

inf_rad_km = 25000

#max_speed is calculated based on the regridding dx but will be limited to a speed of 20 m/s 

n_leadtimes = int(fore_len / timestep)

# Read in lightning data before looping over domains

# Load source lightning grid from file

print("Reading lightning data")

L_source_grid_fil = Path(os.getcwd()+"/W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+LI-2-AF--FD--CHK-BODY--DIS-NC4E_C_EUMT_20261023134736_L2PF_OPE_20261015235000_20261015235030_N__O_0144_0001.nc")

initial_scn = Scene(filenames=[L_source_grid_fil], reader="li_l2_nc")
initial_scn.load(["flash_accumulation"])
L_source_lat_lon = np.array(initial_scn["flash_accumulation"].attrs['area'].get_lonlats())
L_source_lat = L_source_lat_lon[1]
L_source_lon = L_source_lat_lon[0]

L_source_shape = np.shape(L_source_lat)

L_flag = True

try:
    L = read_and_accumulate_LI("flash_accumulation", L_source_shape, LI_indir, LI_accum_period=10, latest_time=date, num_obs=3)

    L_src_grid = xr.Dataset(
        {
            "lat": (["y", "x"], L_source_lat),
            "lon": (["y", "x"], L_source_lon),
        }
    )
    L_src_grid["lat"] = L_src_grid.lat.where(np.isfinite(L_source_lat))
    L_src_grid["lon"] = L_src_grid.lon.where(np.isfinite(L_source_lon))

except:
    print("MTG LI data unavailable - using rainfall-only extrapolation")
    L_flag = False

# Read domains
print("Read domains information")

domain_file = Path(os.getcwd()) / "domains.json"
domains = read_domains(domain_file)
 
for domain_name, domain_info in domains.items():
    
    print(f"Working on domain: {domain_name}")

    centre_lat = domain_info["centre_lat"]
    centre_lon = domain_info["centre_lon"]
    width_px   = domain_info["width_px"]
    height_px  = domain_info["height_px"]
 

    print("")
    print("=" * 80)
    print(f"PROCESSING DOMAIN: {domain_name}")
    print("=" * 80)

    # Find previous RoA files

    fns = []

    missing_file = False

    print("Identifying RoA files")

    for i in range(3, -1, -1):

        t = date - timedelta(minutes=i * timestep)

        fn = (
            R_indir
            / t.strftime("%Y%m%d")
            / f"RoA_{t.strftime('%Y%m%d%H%M')}_{domain_name}.nc"
        )

        if not fn.exists():

            print(f"Missing file: {fn}")

            missing_file = True
            break

        fns.append(str(fn))

    if missing_file:

        print(f"Skipping {domain_name}")
        continue


    print("Reading lat and lon and coordinate values from latest file")
    R_nc = Dataset(fns[-1], "r")
    R_lat = R_nc.variables["latitude"][:]
    R_lon = R_nc.variables["longitude"][:]
    R_x = R_nc.variables["x"]
    R_y = R_nc.variables["y"]
    R_tim = R_nc.variables["time"][0]

    #--------------------------
    # Clean inf values
    #--------------------------

    fill = -9999.0

    # Replace NaN and +/-inf with fill value
    R_lat_clean = np.where(np.isfinite(R_lat), R_lat, fill)
    R_lon_clean = np.where(np.isfinite(R_lon), R_lon, fill)
    R_earth_mask = np.isfinite(R_lat) & np.isfinite(R_lon)

    # Load source RoA grid from file

    R_source_shape = np.shape(R_lat_clean)
    R_input_files = fns

    R = read_RoA("precipitation", R_source_shape, R_input_files)

    # source grid
    R_src_grid = xr.Dataset(
        {
            "lat": (["y", "x"], R_lat),
            "lon": (["y", "x"], R_lon),
        }
    )

    R_src_grid["lat"] = R_src_grid.lat.where(np.isfinite(R_lat))
    R_src_grid["lon"] = R_src_grid.lon.where(np.isfinite(R_lon))

# create cartesian destination grid (better for pysteps extrapolations

    print("Create cartesian grid from domain information")

    proj = Proj(
        proj="laea",
        lat_0=centre_lat,
        lon_0=centre_lon,
        datum="WGS84"
    )


    dx = 4000

    half_width_m  = width_px * dx / 2
    half_height_m = height_px * dx / 2

    x_regrid = np.arange(
        -half_width_m,
         half_width_m + dx,
         dx
    )

    y_regrid = np.arange(
        -half_height_m,
         half_height_m + dx,
         dx
    )

#    x_regrid = np.arange(-5500000, 5500000, dx)
#    y_regrid = np.arange(-5500000, 5500000, dx)

    xx, yy = np.meshgrid(x_regrid, y_regrid)

    lon_regrid, lat_regrid = proj(xx, yy, inverse=True)

    dst_grid = xr.Dataset(
        {
            "lat": (["y", "x"], lat_regrid),
            "lon": (["y", "x"], lon_regrid),
        }
    )

    print("Generate domain hash")
    domain_hash = get_domain_hash(domain_name, domain_info, dx)

    print("Create weights files if required")

# Lightning Forward regridder
    if L_flag:

#        grid_hash = get_grid_hash(L_src_grid,dst_grid)
#        weights_file = Path(os.getcwd()+ f"/regrid_weights/mtg_to_cartesian_{grid_hash}.nc")
        weights_file = Path(os.getcwd()+ f"/regrid_weights/mtg_to_cartesian_{domain_hash}.nc")

        try:
            L_regrid_fwd = xe.Regridder(L_src_grid,dst_grid,"bilinear",filename=weights_file,reuse_weights=True)
        except:
            L_regrid_fwd = xe.Regridder(L_src_grid,dst_grid,"bilinear",filename=weights_file,reuse_weights=False)

# RoA Forward regridder

#    grid_hash = get_grid_hash(R_src_grid,dst_grid)
#    weights_file = Path(os.getcwd()+ f"/regrid_weights/msg_to_cartesian_{grid_hash}.nc")
    weights_file = Path(os.getcwd()+ f"/regrid_weights/msg_to_cartesian_{domain_hash}.nc")

    try:
        R_regrid_fwd = xe.Regridder(R_src_grid,dst_grid,"bilinear",filename=weights_file,reuse_weights=True)
    except:
        R_regrid_fwd = xe.Regridder(R_src_grid,dst_grid,"bilinear",filename=weights_file,reuse_weights=False)

# RoA Reverse regridder

#    grid_hash = get_grid_hash(dst_grid, R_src_grid)
#    weights_file = Path(os.getcwd()+ f"/regrid_weights/cartesian_to_msg_{grid_hash}.nc")
    weights_file = Path(os.getcwd()+ f"/regrid_weights/cartesian_to_msg_{domain_hash}.nc")

    try:
        R_regrid_back = xe.Regridder(dst_grid,R_src_grid,"bilinear",filename=weights_file,reuse_weights=True)
    except:
        R_regrid_back = xe.Regridder(dst_grid,R_src_grid,"bilinear",filename=weights_file,reuse_weights=False)

# LI Reverse regridder
    if L_flag:

#        grid_hash = get_grid_hash(dst_grid, L_src_grid)
#        weights_file = Path(os.getcwd()+ f"/regrid_weights/cartesian_to_mtg_{grid_hash}.nc")
        weights_file = Path(os.getcwd()+ f"/regrid_weights/cartesian_to_mtg_{domain_hash}.nc")

        try:
            L_regrid_back = xe.Regridder(dst_grid,L_src_grid,"bilinear",filename=weights_file,reuse_weights=True)
        except:
            L_regrid_back = xe.Regridder(dst_grid,L_src_grid,"bilinear",filename=weights_file,reuse_weights=False)

# Regrid Lightning to cartesian grid (best for use with pysteps)
        print("Regrid lightning data")

        L_cart = L_regrid_fwd(L)
        L_cart_mask = np.any(np.isfinite(L_cart), axis =0)

# Regrid rainfall to cartesian grid (best for use with pysteps)
    print("regrid RoA data")

    R_cart = R_regrid_fwd(R)
    R_cart_mask = np.any(np.isfinite(R_cart), axis=0)

    print("Build influence fields")
    if L_flag:
        #Build Lightning influence field
        L_cart_mask_hi = L_cart[-1] > 0.05
        distance = distance_transform_edt(~L_cart_mask_hi)
        L_inf_rad_km = 18000
        influence_radius = L_inf_rad_km/dx
        lightning_weight = np.exp(-(distance / influence_radius)**2)
        lightning_weight = gaussian_filter(lightning_weight, 10)
        lightning_weight = lightning_weight*alpha

    #Build rainfall influence field
    R_cart_mask_hi = R_cart[-1] > 10.0
    distance = distance_transform_edt(~R_cart_mask_hi)
    influence_radius = inf_rad_km/dx
    rain_weight = np.exp(-(distance / influence_radius)**2)
    rain_weight = gaussian_filter(rain_weight, 10)
    rain_weight = rain_weight*alpha

    # Create global vector field and combine with tiled rainfall vectors
    nx = len(x_regrid)
    ny = len(y_regrid)

    print("Create coarse global vector field")
    R_cart_coarse = zoom(R_cart, (1, 1/coarse_factor, 1/coarse_factor), order=1)
    UV_global_coarse = dense_lucaskanade(R_cart_coarse)
    U_global = zoom(UV_global_coarse[0], coarse_factor, order=1)*coarse_factor
    V_global = zoom(UV_global_coarse[1], coarse_factor, order=1)*coarse_factor
    U_global = U_global[:ny, :nx]
    V_global = V_global[:ny, :nx]

    tiles = generate_tiles(nx, ny, tile_size, overlap)

    if L_flag:
        print("create lightning based tiled vectors")
        # Create tiled vectors based on lightning motion
        L_results = []

        with ThreadPoolExecutor(max_workers=4) as exe:

            futures = [
                exe.submit(compute_tile_motion, L_cart, tile)
                for tile in tiles
            ]

            for f in futures:
                L_results.append(f.result())

        U_L_res = np.zeros((ny, nx))
        V_L_res = np.zeros((ny, nx))
        W_L_res = np.zeros((ny, nx))

        max_speed = 20/((dx/timestep)/60.0)

        # Loop over tiles and filter for instability. If unstable then use the global but if stable then include the tile in the vector generation.
        print("Combine LI tiles and filter for instability")
        for tile, vec in L_results:

            y0, y1, x0, x1 = tile

            u = vec[0]
            v = vec[1]

            speed = np.sqrt(u**2 + v**2)
            L_tile = L_cart[y0:y1, x0:x1]

            reject_tile=False
            if np.nanmean(speed) > max_speed:
                reject_tile = True

            if reject_tile:
                u = U_global[y0:y1, x0:x1]
                v = V_global[y0:y1, x0:x1]

            U_L_res[y0:y1, x0:x1] += u
            V_L_res[y0:y1, x0:x1] += v
            W_L_res[y0:y1, x0:x1] += 1

        U_L_res /= W_L_res
        V_L_res /= W_L_res

        U_L_res = gaussian_filter(U_L_res, sigma=10)
        V_L_res = gaussian_filter(V_L_res, sigma=10)

# Create tiled vectors based on RoA motion
    R_results = []

    print("Create RoA vector tiles")

    with ThreadPoolExecutor(max_workers=4) as exe:

        futures = [
            exe.submit(compute_tile_motion, R_cart, tile)
            for tile in tiles
        ]

        for f in futures:
            R_results.append(f.result())

    U_R_res = np.zeros((ny, nx))
    V_R_res = np.zeros((ny, nx))
    W_R_res = np.zeros((ny, nx))

    max_speed = 20/((dx/timestep)/60.0)

# Loop over tiles and filter for instability. If unstable then use the global but if stable then include the tile int he vector generation.
    print("Combine RoA tiles and filter for instability")
    for tile, vec in R_results:

        y0, y1, x0, x1 = tile

        u = vec[0]
        v = vec[1]

        speed = np.sqrt(u**2 + v**2)
        R_tile = R_cart[y0:y1, x0:x1]

        reject_tile=False
        if np.nanmean(speed) > max_speed:
            reject_tile = True

        if reject_tile:
            u = U_global[y0:y1, x0:x1]
            v = V_global[y0:y1, x0:x1]

        U_R_res[y0:y1, x0:x1] += u
        V_R_res[y0:y1, x0:x1] += v
        W_R_res[y0:y1, x0:x1] += 1

    U_R_res /= W_R_res
    V_R_res /= W_R_res

    U_R_res = gaussian_filter(U_R_res, sigma=10)
    V_R_res = gaussian_filter(V_R_res, sigma=10)

# Combine global vectors with tiles RoA vectors (using the rainfall weight map, theshold of 10 mm/hr for tiled vector use and a distance of 25 km)

    print("Combine all vector fields together")

    U = (1 - rain_weight) * U_global + rain_weight * U_R_res
    V = (1 - rain_weight) * V_global + rain_weight * V_R_res

# Combine RoA vectors (combined global and tiled) with tiled LI vectors (using the Lightning weight map, threshold of 0.05 for tiled vector use and a distance of 18 km)

    if L_flag:
        U = (1 - lightning_weight) * U + lightning_weight * U_L_res
        V = (1 - lightning_weight) * V + lightning_weight * V_L_res

# final smoothing of vector field)
   
    print("Final smoothing")
 
    U = gaussian_filter(U, sigma=2)
    V = gaussian_filter(V, sigma=2)

    U = np.nan_to_num(U, nan=0.0)
    V = np.nan_to_num(V, nan=0.0)


#    # Quick sanity plots of vector fields
#    import matplotlib.pyplot as plt
#    
#    vmin = -5
#    vmax = 5
#    
#    fig, ax = plt.subplots(4, 2, figsize=(10, 8))
#    
#    im0 = ax[0,0].imshow(U_global, cmap="RdBu_r", vmin=vmin, vmax=vmax)
#    ax[0,0].set_title("U_global")
#    plt.colorbar(im0, ax=ax[0,0], shrink=0.8)
#    
#    im1 = ax[0,1].imshow(V_global, cmap="RdBu_r", vmin=vmin, vmax=vmax)
#    ax[0,1].set_title("V_global")
#    plt.colorbar(im1, ax=ax[0,1], shrink=0.8)
#    
#    im2 = ax[1,0].imshow(U_R_res, cmap="RdBu_r", vmin=vmin, vmax=vmax)
#    ax[1,0].set_title("U_R_res")
#    plt.colorbar(im2, ax=ax[1,0], shrink=0.8)
#    
#    im3 = ax[1,1].imshow(V_R_res, cmap="RdBu_r", vmin=vmin, vmax=vmax)
#    ax[1,1].set_title("V_R_res")
#    plt.colorbar(im3, ax=ax[1,1], shrink=0.8)
#    
#    im4 = ax[2,0].imshow(U_L_res, cmap="RdBu_r", vmin=vmin, vmax=vmax)
#    ax[2,0].set_title("U_L_res")
#    plt.colorbar(im4, ax=ax[2,0], shrink=0.8)
#    
#    im5 = ax[2,1].imshow(V_L_res, cmap="RdBu_r", vmin=vmin, vmax=vmax)
#    ax[2,1].set_title("V_L_res")
#    plt.colorbar(im5, ax=ax[2,1], shrink=0.8)
#    
#    im6 = ax[3,0].imshow(rain_weight, cmap="viridis")
#    ax[3,0].set_title("rain_weight")
#    plt.colorbar(im6, ax=ax[3,0], shrink=0.8)
#    
#    im7 = ax[3,1].imshow(lightning_weight, cmap="viridis")
#    ax[3,1].set_title("lightning_weight")
#    plt.colorbar(im7, ax=ax[3,1], shrink=0.8)
#    
#    for a in ax.flat:
#        a.set_xticks([])
#        a.set_yticks([])
#    
#    plt.tight_layout()
#    plt.show()

    # Perform extrapolation of RoA

    print("Use vector field to perform extrapolation")

    extrapolate = nowcasts.get_method("extrapolation")
    R_cart_f_dLK = extrapolate(R_cart[-1], np.stack([U, V]), timesteps=n_leadtimes).astype(np.float32)
    if L_flag:
        L_cart_f_dLK = extrapolate(L_cart[-1], np.stack([U, V]), timesteps=n_leadtimes).astype(np.float32)

        L_msg_grid = R_regrid_back(L_cart)

    R_f_dLK = R_regrid_back(R_cart_f_dLK)
    if L_flag:
        L_f_dLK = R_regrid_back(L_cart_f_dLK)

    mask_msg = R_regrid_back(R_cart_mask.astype(float))
    mask_msg = mask_msg > 0.5

    mask_msg_3d = np.broadcast_to(mask_msg, R_f_dLK.shape)
    R_earth_mask_3d = np.broadcast_to(R_earth_mask, R_f_dLK.shape)

    R_f_dLK[~mask_msg_3d] = np.nan
    R_f_dLK[~R_earth_mask_3d] = np.nan

    missing_data_f_dLK = np.isnan(R_f_dLK).astype(np.uint8)

# Write extrapolation files

    print("Write file")

    for i in np.arange(0,n_leadtimes, 1):

        valid_tim = R_tim + ((i+1)*10*60)
        valid_dt = datetime.utcfromtimestamp(valid_tim)

        R_outfil = (R_outdir / f"RoA_{indate}{intime}_{domain_name}_{str((i+1)*timestep).zfill(3)}.nc")

        # RoA

        ds_out = xr.Dataset(
            data_vars={
                "precipitation": (
                    ["time","y", "x"],
                    np.expand_dims(R_f_dLK[i, :, :].astype(np.float32), axis=0),
                    {
                        "_FillValue": -9999.0,
                        "standard_name": "rainfall_rate",
                        "long_name": "extrapolation of rainfall rate estimate derived from geostationary satellite data",
                        "units": "mm h-1",        # or "kg m-2" if accumulated
                        "coordinates": "latitude longitude",
                        "grid_mapping": "crs",
                    },
                ),
                "missing_data": (
                    ["time", "y", "x"],
                    np.expand_dims(missing_data_f_dLK[i, :, :].astype(np.int16), axis=0),
                    {
                        "_FillValue": np.int16(-9999),
                        "standard_name": "missing_data_flag",
                        "long_name": "Flag to indicate the presence of required MSG input channels",
                        "units" : "None",
                        "coordinates" : "latitude longitude",
                        "grid_mapping": "crs",
                        "dtype" : "bool",
                    },
                ),
            },
            coords={
                "time": (
                    ["time"],
                    [valid_dt],
                    {
                        "standard_name": "time",
                        "long_name": "time of observation",
                    }
                ),
                "x": (["x"], R_x, {
                    "standard_name": "projection_x_coordinate",
                    "long_name": "x coordinate in projected system",
                    "units": "m",
                    "axis": "X",
                }),
                "y": (["y"], R_y, {
                    "standard_name": "projection_y_coordinate",
                    "long_name": "y coordinate in projected system",
                    "units": "m",
                    "axis": "Y",
                }),
                # lat/lon as auxiliary coordinates
                "latitude": (["y", "x"], R_lat, {
                    "standard_name": "latitude",
                    "long_name": "latitude",
                    "units": "degrees_north",
                    "_FillValue": -9999.0

                }),
                "longitude": (["y", "x"], R_lon, {
                    "standard_name": "longitude",
                    "long_name": "longitude",
                    "units": "degrees_east",
                    "_FillValue": -9999.0

                }),
            },

        )

        # CRS variable
        ds_out["crs"] = xr.DataArray(
            0,
            attrs={
                "grid_mapping_name": "geostationary",
                "longitude_of_projection_origin": 0.0,
                "perspective_point_height": 35785831.0,
                "semi_major_axis": 6378169.0,
                "semi_minor_axis": 6356583.8,
                "sweep_angle_axis": "y",
            },
        )

        # Global attributes
        ds_out.attrs = {
            "base_time": datetime.utcfromtimestamp(R_tim).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "title": "Rain over Africa rainfall estimates",
            "institution": "NCAS, University of Leeds",
            "history": f"Created by extrapolation_RoA_LI.py on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "Conventions": "CF-1.8",
        }

        # Time encoding
        ds_out["time"].encoding["units"] = "seconds since 1970-01-01 00:00:00"
        ds_out["time"].encoding["calendar"] = "gregorian"

        # compression
        encoding = {
             "precipitation": {"zlib": True, "complevel": 4},
             "missing_data": {"zlib": True, "complevel": 4},
        }

        ds_out.to_netcdf(R_outfil, encoding=encoding)

