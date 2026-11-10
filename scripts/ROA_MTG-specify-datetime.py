# Import libraries
from pathlib import Path
import numpy as np
# Compatibility for QuantNN with NumPy 2.x
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
import quantnn
import torch
import tqdm
import xarray as xr
import time
import datetime
import sys
import eumdac
import pickle
import zipfile
import os
import json

from satpy import Scene, find_files_and_readers
from pathlib import Path

# RoA library
from roa import (
    QUANTILES,
    prepare_dataset_for_network,
    Tiler
)
#from roa.data import MSGHRIT
from roa.data import SEVIRI_0DEG_AREADEF
from roa.data import SEVIRI_0DEG_PROJ4
from roa.data import satellite_angles
from roa.utils import fci2seviri_generate_nearest_neighbour_linesample_arrays

def read_domains(domain_file):
    with open(domain_file, "r") as f:
        cfg = json.load(f)
 
    return cfg["domains"]

def get_subdomain_slices(
    lat_grid,
    lon_grid,
    centre_lat,
    centre_lon,
    width_px,
    height_px,
):

    ny, nx = lat_grid.shape

    dist2 = (
        (lat_grid - centre_lat) ** 2 +
        (lon_grid - centre_lon) ** 2
    )

    y_centre, x_centre = np.unravel_index(
        np.nanargmin(dist2),
        dist2.shape
    )

    half_w = width_px // 2
    half_h = height_px // 2

    x0 = x_centre - half_w
    x1 = x0 + width_px

    y0 = y_centre - half_h
    y1 = y0 + height_px

    #
    # Shift box when it falls outside image
    #

    if x0 < 0:
        x1 -= x0
        x0 = 0

    if y0 < 0:
        y1 -= y0
        y0 = 0

    if x1 > nx:
        x0 -= (x1 - nx)
        x1 = nx

    if y1 > ny:
        y0 -= (y1 - ny)
        y1 = ny

    #
    # Final safety clipping
    #

    x0 = max(0, x0)
    y0 = max(0, y0)

    x1 = min(nx, x1)
    y1 = min(ny, y1)

    return y0, y1, x0, x1

def load_mtg_as_msg(extract_dir, idx_file, ts_compact):
 
    import pickle
    import numpy as np
    import xarray as xr
    from datetime import datetime, timedelta
 
    from satpy import Scene, find_files_and_readers
 
    # -------------------------------------------------
    # Load FCI -> SEVIRI lookup tables
    # -------------------------------------------------
 
    with open(idx_file, "rb") as f:
        idx_row_map, idx_col_map, mask_invalid = pickle.load(f)
 
    row_idx = idx_row_map.values.astype(np.int64)
    col_idx = idx_col_map.values.astype(np.int64)
 
    # -------------------------------------------------
    # Read FCI files
    # -------------------------------------------------
 
    target_time = datetime.strptime(ts_compact, "%Y%m%d%H%M")

   
    files = find_files_and_readers(
        base_dir=extract_dir,
        reader="fci_l1c_nc",
        start_time=target_time,
        end_time=target_time+ timedelta(minutes=10)
    )
 
    scn = Scene(
        filenames=files,
        reader="fci_l1c_nc"
    )
 
    scn.load([
        "wv_63",
        "wv_73",
        "ir_87",
        "ir_97",
        "ir_105",
        "ir_123",
        "ir_133"
    ])
 
    orbital_parameters = scn["wv_63"].attrs["orbital_parameters"]
 
    # -------------------------------------------------
    # Create MSG-like dataset
    # -------------------------------------------------
 
    ds = xr.Dataset()
 
    channel_map = {
        "WV_062": "wv_63",
        "WV_073": "wv_73",
        "IR_087": "ir_87",
        "IR_097": "ir_97",
        "IR_108": "ir_105",
        "IR_120": "ir_123",
        "IR_134": "ir_133",
    }
 
    for msg_chan, fci_chan in channel_map.items():
 
        arr = scn[fci_chan].values
 
        #
        # Native FCI orientation correction
        #
        arr = np.flipud(np.fliplr(arr))
 
        remapped = arr[
            row_idx,
            col_idx
        ]
 
        remapped = np.where(
            mask_invalid.values,
            np.nan,
            remapped
        )
 
        ds[msg_chan] = (
            ("y", "x"),
            remapped.astype(np.float32)
        )
 
    # -------------------------------------------------
    # MSG coordinate system
    # -------------------------------------------------
 
    ds = ds.assign_coords({
        "x": SEVIRI_0DEG_AREADEF.projection_x_coords,
        "y": SEVIRI_0DEG_AREADEF.projection_y_coords
    })
 
    xg, yg = np.broadcast_arrays(
        ds.x.data.reshape(1, -1),
        ds.y.data.reshape(-1, 1)
    )
 
    lon, lat = (
        SEVIRI_0DEG_AREADEF
        .get_lonlat_from_projection_coordinates(
            xg,
            yg
        )
    )
 
    ds["longitude"] = (
        ("y", "x"),
        lon.astype(np.float32)
    )
 
    ds["latitude"] = (
        ("y", "x"),
        lat.astype(np.float32)
    )
 
    # -------------------------------------------------
    # Satellite zenith
    # -------------------------------------------------
 
    satellite_zenith = satellite_angles(
        lon,
        lat,
        orbital_parameters["satellite_actual_longitude"],
        orbital_parameters["satellite_actual_latitude"],
        orbital_parameters["satellite_actual_altitude"],
        SEVIRI_0DEG_PROJ4["a"],
        SEVIRI_0DEG_PROJ4["a"] * (
            1 - 1 / SEVIRI_0DEG_PROJ4["rf"]
        )
    )[1].astype(np.float32)
 
    ds["satellite_zenith"] = (
        ("y", "x"),
        satellite_zenith
    )
 
    # -------------------------------------------------
    # Missing-data mask
    # -------------------------------------------------
 
    missing_data = np.isnan(
        np.stack(
            [
                ds["WV_062"].values,
                ds["WV_073"].values,
                ds["IR_087"].values,
                ds["IR_097"].values,
                ds["IR_108"].values,
                ds["IR_120"].values,
                ds["IR_134"].values,
            ]
        )
    ).any(axis=0)
 
    ds["missing_data"] = xr.DataArray(
        missing_data.astype(np.int16),
        dims=("y", "x")
    )
 
    return ds

#=========================================================
# MAIN
#=========================================================

# Read input time as an argument

ts_compact = sys.argv[1]

# read domains

domain_file = Path(os.getcwd()) / "domains.json"
domains = read_domains(domain_file)

# Set paths
MTG_FCI_path = Path(os.getcwd()+"/../data/MTG_FCI_input")
RoA_output_path = Path(os.getcwd()+"/../data/roa_retrievals/"+ts_compact[0:8])
model_path = os.getcwd()+"/../roa/data/network_CPU.pt"
model_gpu = torch.jit.load(model_path, map_location=torch.device('cuda'))

MTG_FCI_path.mkdir(parents=True,exist_ok=True)
RoA_output_path.mkdir(parents=True,exist_ok=True)

t0 = time.perf_counter()

try:

    #=========================================================
    # Read data
    #=========================================================

    idx_file = Path(os.getcwd()) / "regrid_weights/fci2seviri_idxarrays.pickle"
 
    if not idx_file.exists():
 
        print("Generating FCI -> SEVIRI remapping lookup table...")
 
        fci2seviri_generate_nearest_neighbour_linesample_arrays(
            str(idx_file)
        )
 
        print("Lookup table generated.")

    ds = load_mtg_as_msg(
        MTG_FCI_path,
        idx_file,
        ts_compact,
        )

    for domain_name, domain in domains.items():

        print(f"\nProcessing domain: {domain_name}")

        y0, y1, x0, x1 = get_subdomain_slices(
            ds.latitude.values,
            ds.longitude.values,
            domain["centre_lat"],
            domain["centre_lon"],
            domain["width_px"],
            domain["height_px"]
        )

        ds_domain = ds.isel(
            y=slice(y0, y1),
            x=slice(x0, x1)
        )

        print(
            f"Domain bounds:"
            f" y={y0}:{y1}"
            f" x={x0}:{x1}"
            f" shape={ds_domain.dims['y']}x{ds_domain.dims['x']}"
        )

        t1 = time.perf_counter()

        # -------------------------
        # Prepare tiles
        # -------------------------
        x = prepare_dataset_for_network(ds_domain)[None, ...]
        tiler = Tiler(x)
        x = torch.stack(
            [tiler.get_tile(i, j) for i in range(tiler.M) for j in range(tiler.N)],
            dim=0
        )
        x = x.view(-1, *x.shape[2:])
        t2 = time.perf_counter()

        # -------------------------
        # Run inference
        # -------------------------
        dtype = torch.float32
        y_hat = torch.full(
            (len(x), QUANTILES.size, *tiler.tile_size),
            torch.nan,
            dtype=dtype,
            device=torch.device('cpu')
        )

        bs = 32 # batch size -- can be 16,or 32, or 64 depending on GPU memory
        for i in range(0, len(x), bs):
            batch = x[i:(i+bs)].cuda(non_blocking=True)
            with torch.no_grad():
                y_hat[i:(i+bs)] = model_gpu(batch).cpu().to(dtype)


        # Assemble tiles
        y_hat = y_hat.view(tiler.M, tiler.N, 1, *y_hat.shape[1:])
        y_hat = tiler.assemble(y_hat).numpy()

        t3 = time.perf_counter()
        # -------------------------
        # Compute output variables
        # -------------------------
        spatial_shape = y_hat.shape[-2:]
        y_hat = y_hat.reshape(1, QUANTILES.size, -1)
        y_mu = np.concatenate([
            quantnn.quantiles.posterior_mean(
                _y_hat, QUANTILES.astype('float32'), quantile_axis=1
            )
            for _y_hat in np.array_split(y_hat, 10, axis=-1)

        ], axis=-1).reshape(1, *spatial_shape)
        t4 = time.perf_counter()


        #--------------------------
        # Clean inf values
        #--------------------------
  
        fill = -9999.0

        lat = ds_domain.latitude.data
        lon = ds_domain.longitude.data

#        missing_data = ds.missing_data.data
        missing_data = ds_domain.missing_data.data[None, :, :]

        # Replace NaN and +/-inf with fill value
        lat_clean = np.where(np.isfinite(lat), lat, fill)
        lon_clean = np.where(np.isfinite(lon), lon, fill)

        # -------------------------
        # Save to NetCDF
        # -------------------------

        #convert time
        dt = datetime.datetime.strptime(ts_compact, "%Y%m%d%H%M")

        ds_out = xr.Dataset(
            data_vars={
                "precipitation": (
                    ["time","y", "x"],
                    y_mu,
                    {
                        "standard_name": "rainfall_rate",
                        "long_name": "rainfall rate estimate derived from geostationary satellite data",
                        "units": "mm h-1",        # or "kg m-2" if accumulated
                        "_FillValue": -9999.0,
                    },
                ),
                "missing_data": (
                    ["time","y", "x"],
                    missing_data,
                    {
                        "standard_name": "missing_data_flag",
                        "long_name": "Flag to indicate the presence of required MSG input channels",
                        "units": "None",
                        "_FillValue": -9999.0,
                    },
                ),
                "crs": (
                    [],
                    0,
                    {
                        "grid_mapping_name": "geostationary",
                        "longitude_of_projection_origin": 0.0,
                        "perspective_point_height": 35785831.0,
                        "semi_major_axis": 6378169.0,
                        "semi_minor_axis": 6356583.8,
                        "sweep_angle_axis": "y"
                    }
                ),
            },
            coords={
                "time": (
                    ["time"],
                    [dt],
                    {
                        "standard_name": "time",
                        "long_name": "time of observation",
                    }
                ),
                "x": (["x"], ds_domain.x.data, {
                    "standard_name": "projection_x_coordinate",
                    "long_name": "x coordinate in projected system",
                    "units": "m",
                    "axis": "X",
                }),
                "y": (["y"], ds_domain.y.data, {
                    "standard_name": "projection_y_coordinate",
                    "long_name": "y coordinate in projected system",
                    "units": "m",
                    "axis": "Y",
                }),
                # lat/lon as auxiliary coordinates
                "latitude": (["y", "x"], lat_clean, {
                    "standard_name": "latitude",
                    "long_name": "latitude",
                    "units": "degrees_north",
                    "_FillValue": -9999.0
                }),
                "longitude": (["y", "x"], lon_clean, {
                    "standard_name": "longitude",
                    "long_name": "longitude",
                    "units": "degrees_east",
                    "_FillValue": -9999.0
                }),
            },
            attrs={
                "title": "Rain over Africa rainfall estimates",
                "institution": "NCAS, University of Leeds",
                "history": "Created by ROAHRIT-specify-datetime.py",
                "Conventions": "CF-1.8"
            },
        )

        ds_out["time"].encoding["units"] = "seconds since 1970-01-01 00:00:00"
        ds_out["time"].encoding["calendar"] = "gregorian"
        ds_out["precipitation"].attrs["coordinates"] = "latitude longitude"
        ds_out["missing_data"].attrs["coordinates"] = "latitude longitude"
        ds_out["precipitation"].attrs["grid_mapping"] = "crs"
        ds_out["missing_data"].attrs["grid_mapping"] = "crs"

        out_path = RoA_output_path / f"RoA_{ts_compact}_{domain_name}.nc"

        ds_out.to_netcdf(out_path)
        print(f"✅ Saved {out_path}")
        t5 = time.perf_counter()

        # -------------------------
        # Report timings
        # -------------------------
        print(f"\n⏱️  Timestep {ts_compact}:")
        print(f"  Data load and  regrid : {t1 - t0:.2f} s")
        print(f"  Prep/tiling   : {t2 - t1:.2f} s")
        print(f"  Inference GPU : {t3 - t2:.2f} s")
        print(f"  Stats compute : {t4 - t3:.2f} s")
        print(f"  Save output   : {t5 - t4:.2f} s")
        print(f"  TOTAL         : {t5 - t0:.2f} s\n")

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"💥 Failed timestep {ts_compact}: {e}")

