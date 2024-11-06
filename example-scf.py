#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 17:45:14 2023

@author: vpremier
"""

import glob
import shutil
from pathlib import Path
import json
import openeo
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import os 
import xarray as xr
from openeo.udf import execute_local_udf
from openeo.processes import and_, is_nan
import sys


# set working directory
curr_dir=os.getcwd()
base_path = curr_dir+os.sep+"results"

if not os.path.exists(base_path):
    os.mkdir(base_path)

# read a shapefile with the AOI
shp_path = r'/mnt/CEPH_PROJECTS/PROSNOW/research_activity/Senales/auxiliary/boundaries/SenalesCatchment/SenalesCatchment.shp'


catchment_outline = gpd.read_file(shp_path)
bbox = catchment_outline.bounds.iloc[0]

# authentication
eoconn = openeo.connect("https://openeo-dev.vito.be")
eoconn.authenticate_oidc()
eoconn.describe_account()

# load the Copernicus fractional snow cover collection
scf = eoconn.load_collection(
    "FRACTIONAL_SNOW_COVER",
    spatial_extent  = {'west':bbox[0],
                       'east':bbox[2],
                       'south':bbox[1],
                       'north':bbox[3],
                       'crs':4326},
    temporal_extent=['2023-08-02','2023-08-15'],
    bands=["FSCTOC"]
)
# scf.download(base_path + os.sep + 'scf_0.nc')



# Resampling: should be done on the modis extent
scf_rsmpl = scf.resample_spatial(resolution=20, projection=32632,
                                        method = "near")
scf_bbox = scf_rsmpl.filter_bbox(west=631910, south=5167310, east=655890, 
                                 north=5184290, crs=32632)
# scf_bbox.download(base_path + os.sep + 'scf_rsmp.nc')


# Example of udf application: binarize the SCF
binarize = openeo.UDF.from_file('udf-binarize.py', 
                                context={"from_parameter": "context"})
scf_binary = scf_bbox.apply(process=binarize, 
                            context={"snowT": 20})
scf_binary_renamed = scf_binary.rename_labels(dimension="bands",
                                              target=["scf"])
#scf_binary.download(base_path + os.sep + 'scf_binary.nc')


# Alternative: band math
# Problem with the unsigned bits
# scf_test = 100.0 * (scf >= 20) * (scf <= 100) + 205.0 * (scf == 205)
# scf_test = is_nan(scf_test)*0
# scf_test.download(base_path + os.sep + 'scf_binary2.nc')



# def mask_valid(data):
#     binary = data.array_element("scf")
#     mask = is_nan(binary)   
#     return mask
 
# mask = scf_binary_renamed.apply(mask_valid)
# scf_binary_masked = scf_binary_renamed.mask(mask,replacement=0)
# scf_binary_masked.download(base_path + os.sep + 'scf_bin_masked.nc')


# UDF to compute SCF
aggregation = openeo.UDF.from_file('udf-scf.py', context={"from_parameter": "context"})
aggregation = Path('udf-scf.py').read_text()
# scf_aggregated = scf_binary.apply(process=aggregation, 
#                                   context={"pixel_ratio": 25})
# scf_aggregated = scf_binary.apply_dimension(dimension="t",
#                                             process=aggregation, 
#                                             context={"pixel_ratio": 25})
# scf_aggregated.download(base_path + os.sep + 'scf_aggregated.nc')

udf_code = Path('udf-scf.py').read_text()

cube_updated = scf_binary.apply_neighborhood(
    lambda data: data.run_udf(udf=udf_code, runtime='Python-Jep', context=dict()),
    size=[
        {'dimension': 'x', 'value': 600, 'unit': 'px'},
        {'dimension': 'y', 'value': 425, 'unit': 'px'}
    ], overlap=[])

cube_updated.download(base_path + os.sep + 'scf.nc')

# cp_test = cube_updated.save_result(format='netCDF')
# job = cp_test.create_job(title='apply_neigh_8')
# job.start_job()



sys.exit()

"""
DEBUGGING
"""

#execute the udf locally
path=r'./results/scf_binary.nc'
array = xr.open_dataset(path,decode_coords="all")


aggregation = openeo.UDF.from_file('udf-scf-local.py', 
                                   context={"from_parameter": "context"})
myudf = Path('udf-scf-local.py').read_text()
output = execute_local_udf(aggregation, path, fmt='netcdf')


output.get_datacube_list()[0].get_array().plot()
np.shape(output.get_datacube_list()[0])
# output.download('scf_test.nc')
# output.metadata()
# udf = output._datacube_list(output.to_dict())