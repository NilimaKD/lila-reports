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

# %% [markdown] id="oxF3T0hRc_yY"
# # Getting started on colab

# %% [markdown] id="gDRal60HmdcV"
# Use [Conda Colab](https://github.com/conda-incubator/condacolab) to install a conda environment for [this project](https://github.com/restlessronin/lila-reports/.enviornment/).

# %% id="7KpdG5B3mFAu"
# !pip install -q condacolab

# %% [markdown]
# The runtime will restart after the environment is installed.

# %% id="Pe7Gj9gtwo86"
import condacolab
condacolab.install_from_url("https://github.com/restlessronin/lila-reports/releases/download/v0.0.1/conda-lila-reports-0.0.1-Linux-x86_64.sh")

# %% [markdown] id="zN4E-m-XnMJy"
# Mount the Google Drive associated with the google account you used to login to colab.

# %% colab={"base_uri": "https://localhost:8080/"} id="1GzYj7h2w7bK" outputId="05f91591-606d-4f2c-b911-8d718b735868"
from google.colab import drive
drive.mount('/content/drive')

# %% [markdown] id="wDLwwwJjnT1s"
# Imports

# %% colab={"base_uri": "https://localhost:8080/"} id="OneOUJuoc_ya" outputId="25e5a3d3-151e-46cf-817c-bf9b587272c9"
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



# %% [markdown] id="8JNgZbxdnWP4"
# Encapsulate path and CRS encoding while reading shapes.

# %% id="ybA-9tdE0vat"
def read_shape_UT(fname):
  return gpd.read_file('/content/drive/MyDrive/LiLa_Nagapattinam/Practice/' + fname).to_crs(epsg = 4326)


# %% [markdown] id="yy0_tFhrngFH"
# Read two shapes and calculate interesection

# %% id="B-B_fjze1jc4"
shapeN = read_shape_UT('Nagapattinam_proj32644.shp')
shapeTN = read_shape_UT('water_TN.shp')
intersection_shape = overlay(shapeN, shapeTN, how='intersection')

print(intersection_shape.shape)
intersection_shape.plot(cmap='Accent')
