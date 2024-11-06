#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 07:06:04 2024

@author: nciapponi
"""
import openeo 
from openeo.udf.debug import inspect
from openeo.udf import XarrayDataCube

import math
import numpy as np 
import xarray as xr 
from skimage.restoration import inpaint
# import cv2

def apply_datacube(cube: XarrayDataCube, 
                   context: dict) -> XarrayDataCube:
    
    """
    UDF to get the hillshade, i.e., a shaded relief map.
    It takes the illumination angles as input, i.e.,
        altitude of the sun (default 315 degrees: the terrain is lit as if the 
                             light source is coming from the upper left) 
        azimuth of the sun (default 45 degrees: the light is pitched 45 degrees
                            between the horizon and directly overhead) 
        
    First, the slope and aspect are calculated by following Horn (1981).
    
    Input and output datacube have the same dimensions.
    """
    
    altitude = 45
    azimuth = 315
    nodataval = 255
    
    # conversion from degrees to radians
    deg2rad = 180/np.pi
    
    # zenith angle of the sun calculated from the altitude
    zenith_rad = (90 - altitude)/deg2rad
    # zenith angle of the sun calculated from the altitude
    azimuth_rad = azimuth/deg2rad
    
    # load the datacube
    array = cube.get_array()
    # inspect(array.dims)
    array = array.transpose("t", "bands", "y", "x")
    input_shape = array.shape

    data_array = array.values[0,0,:,:]
        
    # get the coodinates of the datacube
    xmin = array.coords['x'].min()
    xmax = array.coords['x'].max()
    ymin = array.coords['y'].min()
    ymax = array.coords['y'].max()
    
    coord_x = np.linspace(start = xmin, 
                          stop = xmax,
                          num = array.shape[-2])
    
    coord_y = np.linspace(start = ymin, 
                          stop = ymax,
                          num = array.shape[-1])
    
    # get the datacube spatial resolution
    resolution = array.coords['x'][-1] - array.coords['x'][-2]
    
    # Calculate gradients
    dz_dx = ((data_array[:-2, 2:] + 2 * data_array[1:-1, 2:] + data_array[2:, 2:]) -
            (data_array[:-2, :-2] + 2 * data_array[1:-1, :-2] + data_array[2:, :-2]))/(8 * resolution.values)
    dz_dy = ((data_array[2:, :-2] + 2 * data_array[2:, 1:-1] + data_array[2:, 2:]) -
            (data_array[:-2, :-2] + 2 * data_array[:-2, 1:-1] + data_array[:-2, 2:])) / (8 * resolution.values)
    
  
    # Compute slope and aspect
    slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)) 
    aspect = np.arctan2(dz_dy, -dz_dx)
    aspect = np.where(aspect < np.pi/2, np.pi/2 - aspect, 5*np.pi/2 - aspect)

    # Compute hillshade
    hillshade = (np.cos(zenith_rad) * np.cos(slope) +
                  np.sin(zenith_rad) * np.sin(slope) * np.cos(azimuth_rad - aspect))
    hillshade = np.clip(hillshade, 0, 1)
    

    # Scale to 0-255
    hillshade = nodataval * hillshade
    hillshade[hillshade<0] = 0
    hillshade[slope==0] = 0
    
    # round to integers
    hillshade = np.round(hillshade)
    
    # Create output array and insert hillshade values
    output = np.zeros_like(data_array) + nodataval
    output[1:-1, 1:-1] = hillshade


    # Create a mask for the 255 values
    # mask = (output == 255) | np.isnan(output)
    

    # interpolated_array = cv2.inpaint(array, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


    # Replace the 255 values using biharmonic inpainting
    # interpolated_array = inpaint.inpaint_biharmonic(output, mask)
    # interpolated_array = interpolated_array.astype(np.uint8)
    coord_t = cube.get_array().coords['t'].values
    band = cube.get_array().coords['bands'].values

    
    # predicted datacube: same dimensions as the input datacube
    predicted_cube = xr.DataArray(output.reshape(input_shape),  # Reshape the output to the original shape (bands, y, x)
                                  dims=['t','bands','y', 'x'],                             
                                  coords=dict(t=coord_t, bands=band, x=coord_x, y=coord_y))
    
    ## Reintroduce time and bands dimensions
#     result_xarray = result_xarray.expand_dims(
#         dim={
#         "t": [np.datetime64(str(cube.t.dt.year.values[0]) + "-01-01")], 
#         "bands": ["prediction"],
#     },
# )

    
    return XarrayDataCube(predicted_cube)
 