#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 07:06:04 2024

@author: vpremier
"""
import openeo 
from openeo.udf.debug import inspect
from openeo.udf import XarrayDataCube

import numpy as np 
import xarray as xr 

def apply_datacube(cube: XarrayDataCube, 
                   context: dict) -> XarrayDataCube:
    
    """
    UDF to get the slope by following Horn (1981).
    
    Input and output datacube have the same dimensions.
    """
    

    nodataval = 255
    
    # conversion from degrees to radians
    deg2rad = 180/np.pi
    
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

    slope = slope*deg2rad

    # Create output array and insert hillshade values
    output = np.zeros_like(data_array) + nodataval
    output[1:-1, 1:-1] = slope
    
    
    coord_t = cube.get_array().coords['t'].values
    band = cube.get_array().coords['bands'].values

    
    # predicted datacube: same dimensions as the input datacube
    predicted_cube = xr.DataArray(output.reshape(input_shape),  # Reshape the output to the original shape (bands, y, x)
                                  dims=['t','bands','y', 'x'],                             
                                  coords=dict(t=coord_t, bands=band, x=coord_x, y=coord_y))
    

    
    return XarrayDataCube(predicted_cube)
 