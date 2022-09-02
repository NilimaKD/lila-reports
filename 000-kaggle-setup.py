# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% papermill={"duration": 2.174351, "end_time": "2022-08-18T08:50:28.508577", "exception": false, "start_time": "2022-08-18T08:50:26.334226", "status": "completed"} tags=[]
import os, sys, subprocess,time
import glob

import pandas as pd
import numpy as np

from shapely.ops import unary_union, polygonize
from shapely.geometry import Polygon
from sklearn.cluster import DBSCAN

from shapely.affinity import rotate
from shapely.geometry import LineString, Point, MultiPolygon,MultiLineString 
from shapely.ops import unary_union, polygonize
import shapely.wkt

import geopandas as gpd
from pyproj import CRS
from rtree import index
import rasterio
#import rasterstats as rs
from geopandas import overlay


# %% papermill={"duration": 0.009479, "end_time": "2022-08-18T08:50:28.520473", "exception": false, "start_time": "2022-08-18T08:50:28.510994", "status": "completed"} tags=[]
def read_shape_UT(fname):
  return gpd.read_file('../input/lila-nagapattinam-dist/LiLa_Nagapattinam/Practice/' + fname).to_crs(epsg = 4326)


# %% papermill={"duration": 7.115993, "end_time": "2022-08-18T08:50:35.638312", "exception": false, "start_time": "2022-08-18T08:50:28.522319", "status": "completed"} tags=[]
shapeN = read_shape_UT('Nagapattinam_proj32644.shp')
shapeTN = read_shape_UT('water_TN.shp')
intersection_shape = overlay(shapeN, shapeTN, how='intersection')

print(intersection_shape.shape)
intersection_shape.plot(cmap='Accent')
