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

# %% [markdown]
# # Setup
#
# ## Notebook Environments
#
# ### Colab
#
# Colab initialization requires a kernel restart. So we do it at the beginning of the notebook.

# %%
import os

# %%
_is_colab = 'COLAB_GPU' in os.environ

# %% [markdown]
# Colab only allows one conda environment. We use `conda-colab` to install an anaconda style environment from a constructor style `.sh` file. See the `.environment` sub folder for instructions.
#
# Colab kernel is restarted after the environment is installed.

# %%
if _is_colab:
    # %pip install -q condacolab
    import condacolab
    condacolab.install_from_url("https://github.com/restlessronin/lila-reports/releases/download/v0.0.2/conda-lila-reports-0.0.2-Linux-x86_64.sh")

# %% [markdown]
# Since colab runtime is restarted by the previous cell, we need to reset globals

# %%
import os
_is_colab = 'COLAB_GPU' in os.environ

# %% [markdown]
# In colab we can mount data files directly from google drive.

# %%
if _is_colab:
    from google.colab import drive
    drive.mount('/content/drive')

# %% [markdown]
# ### Kaggle
#
# Kaggle uses Dockerfiles to define environments, so we wind up using `pip` to install missing packages.

# %%
_is_kaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE','')
if _is_kaggle:
    # %pip install -q matplotlib_scalebar

# %% [markdown]
# ### Local
#
# For local notebooks, we use a conda environment created from `.environment/environment.yaml` to run the notebook. No package installation is necessary.

# %%
#Masking Raster with shapefile

# %%
import rasterio
import rasterio.plot
from rasterio.plot import show
import rasterio.mask

import fiona

import geopandas as gpd
from osgeo import ogr, gdal
from osgeo import gdal_array
from osgeo import gdalconst
import geopandas as gpd
from geopandas import overlay
from matplotlib import pyplot
import matplotlib.pyplot as plt

from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.font_manager as fm
from pygc import great_distance
import matplotlib.patches as mpatches



# %%
#Setting path to directory

def get_rooted(stem):
    return "F:/AV Consulting/2022/LiLa/TN Government/Data/LiLa_Nagapattinam/" + stem

def get_in_workdir(stem):
    if _is_kaggle or _is_colab:
        return './' + stem
    else:
        return get_rooted('workdir/' + stem)

def read_raster_UT(stem):
    return rasterio.open(get_rooted(stem))

def read_shpfi_UT(stem):
    return fiona.open(get_rooted(stem))    

def read_gpd_UT(stem):
    return gpd.read_file(get_rooted(stem)) 

def read_rastergdal_UT(stem):
    return gdal.Open(get_rooted(stem))    

def read_df_UT(stem):
    return gpd.read_file(get_rooted(stem)).to_crs(epsg = 4326)

    

# %% [markdown]
# # Rasters

# %% [markdown]
# ## Slope
#
# To visualize raster, information was used from:
#
# https://zia207.github.io/geospatial-python.io/lesson_06_working-with-raster-data.html
#
# https://stackoverflow.com/questions/61327088/rio-plot-show-with-colorbar
#
# https://matplotlib.org/stable/api/axis_api.html
#

# %% [markdown]
# #### Cut raster to boundary of district shapefile

# %%
_shp_dst = read_gpd_UT('Practice/Nagapattinam_proj32644.shp')
raster = read_raster_UT('Supporting_info/DEM_T44PLT_proj32644_filled_slope.tif')
fig, ax = plt.subplots(figsize=(5, 5))
plt.grid(color="grey",linestyle = '--', linewidth = 0.5)
image = rasterio.plot.show(raster, 
                      transform=raster.transform, 
                      ax=ax, 
                      cmap='gray_r', 
                      )
_shp_dst.plot(ax=ax, color='blue')

# %%
#Open Shapefile with fiona

# %%
with read_shpfi_UT('Practice/Nagapattinam_proj32644.shp') as shapefile:
    for feature in shapefile:
        shapes = [feature['geometry']]



# %%
#Coordinate Reference System of Shapefile

read_shpfi_UT('Practice/Nagapattinam_proj32644.shp').crs

# %%
#Coordinate Reference System of Rasterfile

read_raster_UT('Supporting_info/DEM_T44PLT_proj32644_filled_slope.tif').crs

# %% tags=[]
#Open Raster #with rasterio

# %%
with read_raster_UT('Supporting_info/DEM_T44PLT_proj32644_filled_slope.tif') as src:
    out_image, out_transform = rasterio.mask.mask(src, shapes, crop = True)
    out_meta = src.meta

# %%
out_meta.update({
    'driver':'Gtiff',
    'height': out_image.shape[1],
    'width': out_image.shape[2],
    'transform': out_transform,
})    

# %%
with rasterio.open(get_in_workdir('slope_dst.tif'), 'w', **out_meta) as dst:
    dst.write(out_image)

# %%
rasterio.plot.show(read_raster_UT('workdir/slope_dst.tif'))

# %% [markdown]
# Converting raster into epsg 4326 CRS

# %%
#Open raster
_tif_dst_slope = rasterio.open(get_in_workdir('slope_dst.tif'))

# %%
#Show the CRS of source raster
_tif_dst_slope.crs

# %%
#Defining CRS of reprojected raster. Reprojected raster is named _tif_dst_slope_epsg4326
_tif_dst_slope_epsg4326 = {'init': 'EPSG:4326'}

# %%
#Calculate transform array and shape of reprojected raster
from rasterio.warp import calculate_default_transform, reproject, Resampling
transform, width, height = calculate_default_transform(
    _tif_dst_slope.crs, _tif_dst_slope_epsg4326, _tif_dst_slope.width, _tif_dst_slope.height, *_tif_dst_slope.bounds)

_tif_dst_slope.transform
transform

# %%
#Working of the meta for the destination raster
kwargs = _tif_dst_slope.meta.copy()
kwargs.update({
        'crs': _tif_dst_slope_epsg4326,
        'transform': transform,
        'width': width,
        'height': height,
    })

# %%
#Open destination raster
_tif_dst_slope_epsg4326 = rasterio.open(get_in_workdir('slope_dstepsg.tif'), 'w', **kwargs)


# %%
#Reproject and save raster band date
for i in range(1, _tif_dst_slope.count + 1):
    reproject(
        source = rasterio.band(_tif_dst_slope, i),
        destination = rasterio.band(_tif_dst_slope_epsg4326, i),
        src_crs = _tif_dst_slope.crs,
        dst_crs = _tif_dst_slope_epsg4326,
        resampling = Resampling.nearest
    )


# %%
_tif_dst_slope_epsg4326.close()

# %%
_tif_dst_slope_epsg4326 = read_raster_UT('workdir/slope_dstepsg.tif')
show(_tif_dst_slope_epsg4326)

# %%
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
from osgeo import ogr, gdal
from osgeo import gdal_array
from osgeo import gdalconst

# %%
array = _tif_dst_slope_epsg4326.read()
array.shape
#raster shape is given as (band, rows, columns)

# %%
_tif_dst_slope_epsg4326.colorinterp[0]

# %%
_tif_dst_slope_epsg4326_meta =_tif_dst_slope_epsg4326.meta
print(_tif_dst_slope_epsg4326_meta)

# %%
gdal_data = read_rastergdal_UT('workdir/slope_dstepsg.tif')
gdal_band = gdal_data.GetRasterBand(1)
nodataval = gdal_band.GetNoDataValue()

# convert to a numpy array
data_array = gdal_data.ReadAsArray().astype(float)
data_array

# replace missing values if necessary
if np.any(data_array == nodataval):
    data_array[data_array == nodataval] = np.nan
    data_array[data_array == nodataval] = np.nan

# %% [markdown]
# To create color map:
#
# Visit: https://www.delftstack.com/howto/matplotlib/custom-colormap-using-python-matplotlib/
#
# Visit: https://www.youtube.com/watch?v=qk0n-YaKIkY
#
# The line that creates the colormap is: 
#
# name_of_cmap = LinearSegmentedColormap.from_list('name of this color string', colors=['starting color in hexcode','ending color in hexcode'], N=steps in color of which max is 256)
#
# Ex: AVC_color = LinearSegmentedColormap.from_list('testCmap', colors=['#ec6669','#FFFFFF'], N=256)
#
# There can also be a list of colours for the colourmap, like in the example below
#
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.colors import ListedColormap, LinearSegmentedColormap
# import matplotlib.colors
# xdata=[1,2,3,4,5,6,7,8,9,10,11,12]
# ydata=[x*x for x in xdata]
# norm=plt.Normalize(1,150)
# colorlist=["darkorange", "gold", "lawngreen", "lightseagreen"]
# newcmp = LinearSegmentedColormap.from_list('testCmap', colors=colorlist, N=256)
# c=np.linspace(0,300,12)
# plt.scatter(xdata,ydata,c=c, cmap=newcmp, norm=norm)
# plt.colorbar()
# plt.show()

# %% [markdown]
# To check color palette created:
#
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import numpy as np
#
# def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
#     c1=np.array(mpl.colors.to_rgb(c1))
#     c2=np.array(mpl.colors.to_rgb(c2))
#     return mpl.colors.to_hex((1-mix)*c1 + mix*c2)
#
# c1='#ec6669' #AVC colour
# c2='#FFFFFF' #white
# n=500
#
# fig, ax = plt.subplots(figsize=(8, 5))
# for x in range(n+1):
#     ax.axvline(x, color=colorFader(c1,c2,x/n), linewidth=4) 
# plt.show()

# %% [markdown]
# To check raster with a specific colour map: 
#
# show(rasterfile_name, cmap=nameofdefinedcolourmap)

# %%
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors

#Set colour
AVC_grey = LinearSegmentedColormap.from_list('testCmap1', colors=['#F0F0F0', '#c3c3c3', '#969696', '#686868'], N=256)


# %%
#Show min and max of array without nan values
a = np.nanmax(data_array)
b = np.nanmin(data_array)
a, b

# %% [markdown]
# #### Generate image of slope features of district

# %%
import matplotlib.ticker as ticker
from matplotlib.ticker import LinearLocator, FormatStrFormatter


# %%
def plot_cities(fig, ax):
    shp_cities = read_gpd_UT("extrainputs/Mayiladuthurai_major_towns.shp")
    shp_cities['coords'] = shp_cities['geometry'].apply(lambda x: x.representative_point().coords[:])
    shp_cities['coords'] = [coords[0] for coords in shp_cities['coords']]
    shp_cities["coords"].tolist()
    shp_cities[['lat', 'lon', 'zero']] = gpd.GeoDataFrame(shp_cities['coords'].tolist(), index=shp_cities.index)

    x = shp_cities["lat"]
    y = shp_cities["lon"]    
    labels =shp_cities["name"]

    for i in range(0,len(shp_cities)):
        plt.text(x[i]+0.008,y[i]+0.008,labels[i],fontsize=4,color = '#c3c3c3', ha = 'center')

    shp_cities.plot(ax=ax4, markersize=3, color='grey')

def plot_common_features(fig, ax):
    plt.rcParams['font.family'] = 'Helvetica'
    plt.grid(color="grey",linestyle = '--', linewidth = 0.1)

    dx = great_distance(start_latitude=11.1, start_longitude=-79.5, end_latitude=11.1, end_longitude=-79.6)
    scalebar = ScaleBar(dx = 109250.50301657, location ="lower left", frameon=False, color='lightgrey', sep=1.5, width_fraction=0.012)
    ax.add_artist(scalebar)
    scalebar.font_properties.set_size(5.5)
    scalebar.font_properties.set_family('Helvetica')

    ax.tick_params(axis='x', colors='grey', labelsize=3, labeltop = 'True', labelrotation = 270)
    ax.tick_params(axis='y', colors='grey', labelsize=3, labelright = 'True') #reducing the size of the axis values
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f')) #axis value formatting for both axis
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xlim(79.40, 80)
    ax.set_ylim(10.93, 11.45)

    ax.spines['bottom'].set_color('none')
    ax.spines['top'].set_color('none') 
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')

    x, y, arrow_length = 0.5, 0.99, 0.1
    ax.annotate('N',color= "lightgrey", xy=(x, y), xytext=(x, y-arrow_length), arrowprops=dict(facecolor='none',edgecolor="lightgrey", width=4, headwidth=12), ha='center', va='center', fontsize=10, xycoords=ax4.transAxes)


# %%
slope = read_raster_UT('workdir/slope_dstepsg.tif')
_shp_cities = read_gpd_UT("extrainputs/Mayiladuthurai_major_towns.shp")

fig1, ax1 = plt.subplots(figsize=(5, 5))
plt.grid(color="grey",linestyle = '--', linewidth = 0.5)

font = fm.FontProperties('Helvetica')

plt.grid(color="grey",linestyle = '--', linewidth = 0.1)

_shp_cities['coords'] = _shp_cities['geometry'].apply(lambda x: x.representative_point().coords[:])
_shp_cities['coords'] = [coords[0] for coords in _shp_cities['coords']]

_shp_cities["coords"].tolist()
_shp_cities[['lat', 'lon', 'zero']] = gpd.GeoDataFrame(_shp_cities['coords'].tolist(), index=_shp_cities.index)

x = _shp_cities["lat"]
y = _shp_cities["lon"]    
labels =_shp_cities["name"]

for i in range(0,len(_shp_cities)):
    plt.text(x[i]+0.008,y[i]+0.008,labels[i],fontsize=4,color = '#c3c3c3', ha = 'center')
    
_shp_cities.plot(ax=ax1, markersize=3, color='grey')


dx = great_distance(start_latitude=11.1, start_longitude=-79.5, end_latitude=11.1, end_longitude=-79.6)
scalebar = ScaleBar(dx = 109250.50301657, location ="lower left", frameon=False, color='lightgrey', sep=1.5, width_fraction=0.012)
ax1.add_artist(scalebar)
scalebar.font_properties.set_size(5.5)
scalebar.font_properties.set_family('Helvetica')

ax1.tick_params(axis='x', colors='grey', labelsize=3, labeltop = 'True', labelrotation = 270)
ax1.tick_params(axis='y', colors='grey', labelsize=3, labelright = 'True') #reducing the size of the axis values

ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f')) #axis value formatting for both axis
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax1.set_xlim(79.40, 80)
ax1.set_ylim(10.93, 11.45)

ax1.spines['bottom'].set_color('none')
ax1.spines['top'].set_color('none') 
ax1.spines['right'].set_color('none')
ax1.spines['left'].set_color('none')

x, y, arrow_length = 0.5, 0.99, 0.1
ax1.annotate('N',color= "lightgrey", xy=(x, y), xytext=(x, y-arrow_length), arrowprops=dict(facecolor='none',edgecolor="lightgrey", width=4, headwidth=12), ha='center', va='center', fontsize=10, xycoords=ax1.transAxes)


# use imshow so that we have something to map the colorbar to
image_hidden = ax.imshow(data_array, 
                         cmap=AVC_grey, 
                         )


# add colorbar using the now hidden image
cbaxes = fig1.add_axes([0.8, 0.2, 0.05, 0.05], frameon=False, title = 'Legend\n \n Height\n above sea-level (m)') 

cbar = fig1.colorbar(image_hidden, cax=cbaxes)
cbar.ax.tick_params(labelsize=5.5)
cbaxes.title.set_size(5.5)


# plot on the same axis with rasterio.plot.show
image = rasterio.plot.show(slope, 
                      transform=slope.transform, 
                      ax=ax1, 
                      cmap=AVC_grey, 
                      )


plt.savefig(get_in_workdir("Slope.jpg"),dpi =1500)

print(plt.rcParams['font.family'])

# %%
_shp_cities.head()

# %% [markdown]
# ## GHI

# %%
GHI_gdal_data = read_rastergdal_UT('extrainputs/GHIepsg_Nagapattinam.tif')
gdal_band = GHI_gdal_data.GetRasterBand(1)
nodataval = 1.1754943508222875e-38

# convert to a numpy array
GHI_data_array = GHI_gdal_data.ReadAsArray().astype(float)
GHI_data_array

# replace missing values if necessary
if np.any(GHI_data_array == nodataval):
    GHI_data_array[GHI_data_array == nodataval] = np.nan
    GHI_data_array[GHI_data_array == nodataval] = np.nan

# %%
GHI_data_array.shape

# %%
#Show min and max of array without nan values
a = np.nanmax(GHI_data_array)
b = np.nanmin(GHI_data_array)
a, b

# %%
import pandas as pd
df = pd.DataFrame(GHI_data_array)
df.head()

# %%
#Set colour - https://www.schemecolor.com/burgundy-red-orange.php
AVC_color2 = LinearSegmentedColormap.from_list('testCmap', 
                                               colors=['#E9FF70', '#F6DE26', '#FF8811', '#F71A16', '#C4171C', '#A70B0B'], N=256)

# %%
#show(GHI, cmap=AVC_color2)

# %%


# %%
x = great_distance(start_latitude=11.1, start_longitude=79, end_latitude=11.1, end_longitude=80)
print(x)

# %%
GHI = read_raster_UT('extrainputs/GHIepsg_Nagapattinam.tif')

_shp_cities = read_gpd_UT("extrainputs/Mayiladuthurai_major_towns.shp")

fig2, ax2 = plt.subplots(figsize=(5, 5))
plt.grid(color="grey",linestyle = '--', linewidth = 0.5)

plt.rcParams['font.family'] = 'Helvetica'
font = fm.FontProperties('Helvetica')

plt.grid(color="grey",linestyle = '--', linewidth = 0.1)

_shp_cities['coords'] = _shp_cities['geometry'].apply(lambda x: x.representative_point().coords[:])
_shp_cities['coords'] = [coords[0] for coords in _shp_cities['coords']]

_shp_cities["coords"].tolist()
_shp_cities[['lat', 'lon', 'zero']] = gpd.GeoDataFrame(_shp_cities['coords'].tolist(), index=_shp_cities.index)

x = _shp_cities["lat"]
y = _shp_cities["lon"]    
labels =_shp_cities["name"]

for i in range(0,len(_shp_cities)):
    plt.text(x[i]+0.008,y[i]+0.008,labels[i],fontsize=4,color = '#c3c3c3', ha = 'center')
    
_shp_cities.plot(ax=ax2, markersize=3, color='grey')


dx = great_distance(start_latitude=11.1, start_longitude=-79.5, end_latitude=11.1, end_longitude=-79.6)
scalebar = ScaleBar(dx = 109250.50301657, location ="lower left", frameon=False, color='lightgrey', sep=1.5, width_fraction=0.012)
ax2.add_artist(scalebar)
scalebar.font_properties.set_size(5.5)
scalebar.font_properties.set_family('Helvetica')

ax2.tick_params(axis='x', colors='grey', labelsize=3, labeltop = 'True', labelrotation = 270)
ax2.tick_params(axis='y', colors='grey', labelsize=3, labelright = 'True') #reducing the size of the axis values

ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f')) #axis value formatting for both axis
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax2.set_xlim(79.40, 80)
ax2.set_ylim(10.93, 11.45)

ax2.spines['bottom'].set_color('none')
ax2.spines['top'].set_color('none') 
ax2.spines['right'].set_color('none')
ax2.spines['left'].set_color('none')

x, y, arrow_length = 0.5, 0.99, 0.1
ax2.annotate('N',color= "lightgrey", xy=(x, y), xytext=(x, y-arrow_length), arrowprops=dict(facecolor='none',edgecolor="lightgrey", width=4, headwidth=12), ha='center', va='center', fontsize=10, xycoords=ax2.transAxes)


# use imshow so that we have something to map the colorbar to
image_hidden = ax.imshow(GHI_data_array, cmap=AVC_color2, vmin = 1800, vmax = 2275)

# add colorbar using the now hidden image
cbaxes = fig2.add_axes([0.8, 0.2, 0.05, 0.05], frameon=False, title = 'Legend\n \n GHI\n kwh/m2/year') 

cbar = fig2.colorbar(image_hidden, cax=cbaxes)
cbar.ax.tick_params(labelsize=5)
cbaxes.title.set_size(5.5)

# plot on the same axis with rasterio.plot.show
image = rasterio.plot.show(GHI, 
                      transform=GHI.transform, 
                      ax=ax2, 
                      cmap=AVC_color2, 
                      vmin = 1800,
                      vmax = 2275,
                      )

plt.savefig(get_in_workdir("GHI.jpg"),dpi =1500)

print(plt.rcParams['font.family'])

# %% [markdown]
# # Shapefiles

# %% [markdown]
# ## Water Bodies 
# Rivers and Lakes

# %%
_shp_water = read_gpd_UT("Practice/water_TN.shp")
_shp_district = read_df_UT('Practice/Nagapattinam_proj32644.shp')

_shp_water.plot()

# %%
_shp_dst_water = overlay(_shp_water,_shp_district,how ="intersection")
_shp_dst_water.plot()

# %% [markdown]
# This plot shows the water bodies within the boundaries of the district.
# Details of the library can be found here : https://geopandas.org/en/stable/docs/user_guide/mapping.html

# %%
_shp_dst_water["color"] = "Waterbodies"

_shp_dst_water["GEOMORPHOL"].unique()

# %%
plt.rcParams['font.family'] = 'Helvetica'

_shp_cities = read_gpd_UT("extrainputs/Mayiladuthurai_major_towns.shp")

fig3, ax3 = plt.subplots(figsize=(5, 5))
plt.grid(color="grey",linestyle = '--', linewidth = 0.5)

plt.rcParams['font.family'] = 'Helvetica'
font = fm.FontProperties('Helvetica')

plt.grid(color="grey",linestyle = '--', linewidth = 0.1)

_shp_cities['coords'] = _shp_cities['geometry'].apply(lambda x: x.representative_point().coords[:])
_shp_cities['coords'] = [coords[0] for coords in _shp_cities['coords']]

_shp_cities["coords"].tolist()
_shp_cities[['lat', 'lon', 'zero']] = gpd.GeoDataFrame(_shp_cities['coords'].tolist(), index=_shp_cities.index)

x = _shp_cities["lat"]
y = _shp_cities["lon"]    
labels =_shp_cities["name"]

for i in range(0,len(_shp_cities)):
    plt.text(x[i]+0.008,y[i]+0.008,labels[i],fontsize=4,color = '#c3c3c3', ha = 'center')
    

dx = great_distance(start_latitude=11.1, start_longitude=-79.5, end_latitude=11.1, end_longitude=-79.6)
scalebar = ScaleBar(dx = 109250.50301657, location ="lower left", frameon=False, color='lightgrey', sep=1.5, width_fraction=0.012)
ax3.add_artist(scalebar)
scalebar.font_properties.set_size(5.5)
scalebar.font_properties.set_family('Helvetica')

ax3.tick_params(axis='x', colors='grey', labelsize=3, labeltop = 'True', labelrotation = 270)
ax3.tick_params(axis='y', colors='grey', labelsize=3, labelright = 'True') #reducing the size of the axis values

ax3.xaxis.set_major_formatter(FormatStrFormatter('%.2f')) #axis value formatting for both axis
ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax3.set_xlim(79.40, 80)
ax3.set_ylim(10.93, 11.45)

ax3.spines['bottom'].set_color('none')
ax3.spines['top'].set_color('none') 
ax3.spines['right'].set_color('none')
ax3.spines['left'].set_color('none')

x, y, arrow_length = 0.5, 0.99, 0.1
ax3.annotate('N',color= "lightgrey", xy=(x, y), xytext=(x, y-arrow_length), arrowprops=dict(facecolor='none',edgecolor="lightgrey", width=4, headwidth=12), ha='center', va='center', fontsize=10, xycoords=ax3.transAxes)

_shp_dst_waterbodies = _shp_dst_water.plot(figsize =(5,5), color='#00B9F2', categorical=True, ax=ax3)
_shp_cities.plot(ax=ax3, markersize=3, color='grey')
_plt_district = _shp_district.plot(ax=ax3, figsize =(5,5),color="none",linewidth = 0.5)

blue_patch = mpatches.Patch(color='#00B9F2', label='Waterbodies')
    
plt.legend(handles = [blue_patch], loc = 'upper left', bbox_to_anchor=(0.8, 0.2), title = 'Legend\n', fontsize = 5.5, markerscale = 2, title_fontsize = 5.5, framealpha= 0, borderpad = 0.3, handletextpad = 0.5, handlelength = 1.0)


print(plt.rcParams['font.family'])
plt.savefig(get_in_workdir("water_rivers_lakes.jpg"),dpi =1500)
plt.show()

# %% [markdown]
# All Water Bodies

# %%
shp_land_cover = read_df_UT('extrainputs/LC_mayil_shapefile.shp')
shp_land_cover["DN"].unique()

# %%
shp_land_cover_builtup = shp_land_cover[shp_land_cover["DN"] == 6 ]
shp_land_cover_Barren = shp_land_cover[shp_land_cover["DN"] == 1 ]
shp_land_cover_sparseveg = shp_land_cover[shp_land_cover["DN"] == 2 ]
shp_land_cover_cropland  = shp_land_cover[shp_land_cover["DN"] == 3 ]
shp_land_cover_Forest = shp_land_cover[shp_land_cover["DN"] == 4 ]
shp_land_cover_Water = shp_land_cover[shp_land_cover["DN"] == 5 ]
shp_land_cover_Water1 = shp_land_cover[shp_land_cover["DN"] == 50 ]

# %%
plt.rcParams['font.family'] = 'Helvetica'

_shp_cities = read_gpd_UT("extrainputs/Mayiladuthurai_major_towns.shp")

fig55, ax55 = plt.subplots(figsize=(5, 5))
plt.grid(color="grey",linestyle = '--', linewidth = 0.5)

plt.rcParams['font.family'] = 'Helvetica'
font = fm.FontProperties('Helvetica')

plt.grid(color="grey",linestyle = '--', linewidth = 0.1)

_shp_cities['coords'] = _shp_cities['geometry'].apply(lambda x: x.representative_point().coords[:])
_shp_cities['coords'] = [coords[0] for coords in _shp_cities['coords']]

_shp_cities["coords"].tolist()
_shp_cities[['lat', 'lon', 'zero']] = gpd.GeoDataFrame(_shp_cities['coords'].tolist(), index=_shp_cities.index)

x = _shp_cities["lat"]
y = _shp_cities["lon"]    
labels =_shp_cities["name"]

for i in range(0,len(_shp_cities)):
    plt.text(x[i]+0.008,y[i]+0.008,labels[i],fontsize=4,color = '#c3c3c3', ha = 'center')
    

dx = great_distance(start_latitude=11.1, start_longitude=-79.5, end_latitude=11.1, end_longitude=-79.6)
scalebar = ScaleBar(dx = 109250.50301657, location ="lower left", frameon=False, color='lightgrey', sep=1.5, width_fraction=0.012)
ax55.add_artist(scalebar)
scalebar.font_properties.set_size(5.5)
scalebar.font_properties.set_family('Helvetica')

ax55.tick_params(axis='x', colors='grey', labelsize=3, labeltop = 'True', labelrotation = 270)
ax55.tick_params(axis='y', colors='grey', labelsize=3, labelright = 'True') #reducing the size of the axis values

ax55.xaxis.set_major_formatter(FormatStrFormatter('%.2f')) #axis value formatting for both axis
ax55.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax55.set_xlim(79.40, 80)
ax55.set_ylim(10.93, 11.45)

ax55.spines['bottom'].set_color('none')
ax55.spines['top'].set_color('none') 
ax55.spines['right'].set_color('none')
ax55.spines['left'].set_color('none')

x, y, arrow_length = 0.5, 0.99, 0.1
ax55.annotate('N',color= "lightgrey", xy=(x, y), xytext=(x, y-arrow_length), arrowprops=dict(facecolor='none',edgecolor="lightgrey", width=4, headwidth=12), ha='center', va='center', fontsize=10, xycoords=ax55.transAxes)


_plt_district = _shp_district.plot(ax=ax55, figsize =(5,5),color="none",linewidth = 0.5)

_shp_cities.plot(ax=ax55, markersize=3, color='grey', zorder = 1)
#shp_land_cover_Barren.plot(color="#424242",ax =ax7, label ='Unused')
#shp_land_cover_sparseveg.plot(color="#f096ff",ax =ax7, label = 'Sparse vegetation')
#shp_land_cover_cropland.plot(color="#18a558",ax =ax7, label = 'Cropland', zorder = 0)
#shp_land_cover_Forest.plot(color="#B1D355",ax =ax7, label = 'Tree cover')
#shp_land_cover_builtup.plot(color="#e73429",ax =ax7, label = 'Built-up')
shp_land_cover_Water.plot(color="#00B9F2",ax =ax55, label = 'Water')
shp_land_cover_Water1.plot(color="#00B9F2",ax =ax55)

#ax7.legend(loc='upper left', bbox_to_anchor=(0.8, 0.2), title = 'Legend\n', fontsize = 5.5, markerscale = 2, title_fontsize = 5.5, framealpha= 0, borderpad = 0.3, handletextpad = 0.5, handlelength = 1.0)

#Unused = mpatches.Patch(color="#424242", label ='Unused')
#SpV = mpatches.Patch(color="#f096ff", label = 'Sparse vegetation')
#Crop = mpatches.Patch(color="#18a558", label = 'Cropland')
#TC = mpatches.Patch(color="#B1D355", label = 'Tree cover')
#BU = mpatches.Patch(color="#e73429", label = 'Built-up')
Water = mpatches.Patch(color="#00B9F2", label = 'Water')

    
plt.legend(handles = [Water], loc = 'upper left', bbox_to_anchor=(0.8, 0.2), title = 'Legend\n', fontsize = 5.5, markerscale = 2, title_fontsize = 5.5, framealpha= 0, borderpad = 0.3, handletextpad = 0.5, handlelength = 1.0)


plt.savefig(get_in_workdir("Waterbodies.jpg"),dpi =1500)
plt.show()

# %% [markdown]
# ## Accessibility

# %% [markdown]
# ### Primary and Secondary Roads

# %%
_shp_roads = read_df_UT("Supporting_info/output_osmroad_edges_11.776643009779821_10.743913945502888_80.19273383153288_79.14689901832789.geojson/edges.shp")

#_shp_roads.plot()

# %%
_shp_roads.shape
_shp_roads["highway"].unique()

# %%
_shp_roads_secondary = _shp_roads.apply(lambda row: row[_shp_roads['highway'].isin(['secondary'])])
_shp_roads_primary = _shp_roads.apply(lambda row: row[_shp_roads['highway'].isin(['primary'])])

_shp_dst_roads_secondary = overlay(_shp_roads_secondary,_shp_district,how ="intersection")
_shp_dst_roads_primary = overlay(_shp_roads_primary,_shp_district,how ="intersection")

# %%
_shp_roads_primary_secondary = gpd.pd.concat([_shp_dst_roads_primary, _shp_dst_roads_secondary])
_shp_roads_primary_secondary.to_file(get_in_workdir('roads_prim_sec.shp'))

# %% [markdown]
# ### Railways

# %%
_shp_railways = read_df_UT("Supporting_info/railway/Railways.shp")
#_shp_railways.plot()

# %%
_shp_dst_railways = overlay(_shp_railways,_shp_district,how ="intersection")

# %%
_shp_dst_railways.to_file(get_in_workdir('railsways_dst.shp'))

# %%
_shp_dst_roads_secondary.plot()
_shp_dst_roads_primary.plot()
_shp_dst_railways.plot()

# %% [markdown]
# ### Accessibility (Composite)
#
# This plot layers the primary and secondary roads with the railway lines in the district. Details of the library can be found here : https://geopandas.org/en/stable/docs/user_guide/mapping.html

# %%
fig4, ax4 = plt.subplots(figsize=(5, 5))

plot_common_features(fig4, ax4)

_plt_district = _shp_district.plot(ax=ax4, figsize =(5,5),color="none",linewidth = 0.5)
_shp_dst_roads_secondary.plot(color="#f0b8b3",label ="Secondary roads",ax=ax4, linewidth=0.5)
_shp_dst_roads_primary.plot(color="#df2a29",label ="Primary roads",ax=ax4, linewidth=0.5)
_shp_dst_railways.plot(color="#da0404",label ="Railway roads",ax=ax4, linestyle='--', linewidth=0.5)

plot_cities(fig4, ax4)

ax4.legend(loc='upper left', bbox_to_anchor=(0.8, 0.2), title = 'Legend\n', fontsize = 5.5, markerscale = 2, title_fontsize = 5.5, framealpha= 0, borderpad = 0.3, handletextpad = 0.5, handlelength = 1.0)

plt.savefig(get_in_workdir("Accessibility.jpg"),dpi =1500)
plt.show()

# %% [markdown]
# ## Evacuation

# %% [markdown]
# ### Powerlines

# %%
_shp_powerlines = read_df_UT("Supporting_info/osm_powline_11.776643009779821_10.743913945502888_80.19273383153288_79.14689901832789.geojson")

# %%
_shp_powerlines.info()
_shp_powerlines.plot()

# %%
_shp_dst_powerlines = overlay(_shp_powerlines, _shp_district, how='intersection')
#_shp_dst_powerlines.info()

# %%

# %% [markdown]
# ### Substations

# %%
_shp_substations = read_df_UT("Supporting_info/list_substation_TN_corr.geojson/list_substation_TN_corr.shp")
_shp_dst_substations = overlay(_shp_substations, _shp_district, how='intersection')
_shp_dst_substations["geometry"]
_shp_dst_substations.head()

# %%

# %% [markdown]
# ### Evacuation
#
# This plot layers the powerlines and substation locations in the district.
# Details of the library can be found here : https://geopandas.org/en/stable/docs/user_guide/mapping.html

# %%
fig5, ax5 = plt.subplots(figsize=(5, 5))
plt.grid(color="grey",linestyle = '--', linewidth = 0.5)

plt.rcParams['font.family'] = 'Helvetica'
font = fm.FontProperties('Helvetica')
fontprops = fm.FontProperties(size=5.5)

plt.grid(color="grey",linestyle = '--', linewidth = 0.1)

#Voltage annotation
_shp_dst_powerlines['coords'] = _shp_dst_powerlines['geometry'].apply(lambda x: x.representative_point().coords[:])
_shp_dst_powerlines['coords'] = [coords[0] for coords in _shp_dst_powerlines['coords']]
 
    
_shp_dst_powerlines["coords"].tolist()
_shp_dst_powerlines[['lat', 'lon']] = gpd.GeoDataFrame(_shp_dst_powerlines['coords'].tolist(), index=_shp_dst_powerlines.index)

x = _shp_dst_powerlines["lat"]
y = _shp_dst_powerlines["lon"]    
labels =_shp_dst_powerlines["voltage"]

for i in range(0,len(_shp_dst_powerlines), 4):
    plt.text(x[i],y[i]+0.002,labels[i],fontsize=1.5,color = "red")


#Substations
_shp_dst_substations['coords'] = _shp_dst_substations['geometry'].apply(lambda x: x.representative_point().coords[:])
_shp_dst_substations['coords'] = [coords[0] for coords in _shp_dst_substations['coords']]

_shp_dst_substations["coords"].tolist()
_shp_dst_substations[['lat', 'lon']] = gpd.GeoDataFrame(_shp_dst_substations['coords'].tolist(), index=_shp_dst_substations.index)

x = _shp_dst_substations["lat"]
y = _shp_dst_substations["lon"]    
labels =_shp_dst_substations["Capacity i"]

for i in range(0,len(_shp_dst_substations)):
    plt.text(x[i]+0.002,y[i],labels[i],fontsize=1.5,color = 'red', ha = 'left')
    

#Major towns and cities
_shp_cities['coords'] = _shp_cities['geometry'].apply(lambda x: x.representative_point().coords[:])
_shp_cities['coords'] = [coords[0] for coords in _shp_cities['coords']]

_shp_cities["coords"].tolist()
_shp_cities[['lat', 'lon', 'zero']] = gpd.GeoDataFrame(_shp_cities['coords'].tolist(), index=_shp_cities.index)

x = _shp_cities["lat"]
y = _shp_cities["lon"]    
labels =_shp_cities["name"]

for i in range(0,len(_shp_cities)):
    plt.text(x[i]+0.008,y[i]+0.008,labels[i],fontsize=4,color = '#c3c3c3', ha = 'center')



#Graph features
dx = great_distance(start_latitude=11.1, start_longitude=-79.5, end_latitude=11.1, end_longitude=-79.6)
scalebar = ScaleBar(dx = 109250.50301657, location ="lower left", frameon=False, color='lightgrey', sep=1.5, width_fraction=0.012)
ax5.add_artist(scalebar)
scalebar.font_properties.set_size(5.5)
scalebar.font_properties.set_family('Helvetica')

ax5.tick_params(axis='x', colors='grey', labelsize=3, labeltop = 'True', labelrotation = 270)
ax5.tick_params(axis='y', colors='grey', labelsize=3, labelright = 'True') #reducing the size of the axis values

ax5.xaxis.set_major_formatter(FormatStrFormatter('%.2f')) #axis value formatting for both axis
ax5.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax5.set_xlim(79.40, 80)
ax5.set_ylim(10.93, 11.45)

ax5.spines['bottom'].set_color('none')
ax5.spines['top'].set_color('none') 
ax5.spines['right'].set_color('none')
ax5.spines['left'].set_color('none')

x, y, arrow_length = 0.5, 0.99, 0.1
ax5.annotate('N',color= "lightgrey", xy=(x, y), xytext=(x, y-arrow_length), arrowprops=dict(facecolor='none',edgecolor="lightgrey", width=4, headwidth=12), ha='center', va='center', fontsize=10, xycoords=ax5.transAxes)

#plots
_plt_district = _shp_district.plot(ax=ax5, figsize =(5,5),color="none",linewidth = 0.5)
_shp_dst_substations.plot(color='red', marker='x', markersize=2.5, ax=ax5, linewidth=0.05, label='Substations')
_shp_dst_powerlines.plot(color="#dd2c0e",ax =ax5, linewidth=0.3, linestyle='--', label='Transmission') 
_shp_cities.plot(ax=ax5, markersize=3, color='grey')

ax5.legend(loc='upper left', bbox_to_anchor=(0.8, 0.2), title = 'Legend\n', fontsize = 5.5, markerscale = 2, title_fontsize = 5.5, framealpha= 0, borderpad = 0.3, handletextpad = 0.5, handlelength = 1.0)


print(plt.rcParams['font.family'])

plt.savefig(get_in_workdir("powerlines.jpg"),dpi =1500) 
plt.show()

# %% [markdown]
# ## Built-up Area

# %%
shp_land_cover = read_df_UT('extrainputs/LC_mayil_shapefile.shp')
shp_land_cover["DN"].unique()

# %%
shp_land_cover_builtup = shp_land_cover[shp_land_cover["DN"] == 6 ]
shp_land_cover_Barren = shp_land_cover[shp_land_cover["DN"] == 1 ]
shp_land_cover_sparseveg = shp_land_cover[shp_land_cover["DN"] == 2 ]
shp_land_cover_cropland  = shp_land_cover[shp_land_cover["DN"] == 3 ]
shp_land_cover_Forest = shp_land_cover[shp_land_cover["DN"] == 4 ]
shp_land_cover_Water = shp_land_cover[shp_land_cover["DN"] == 5 ]
shp_land_cover_Water1 = shp_land_cover[shp_land_cover["DN"] == 50 ]

# %%
plt.rcParams['font.family'] = 'Helvetica'

_shp_cities = read_gpd_UT("extrainputs/Mayiladuthurai_major_towns.shp")

fig6, ax6 = plt.subplots(figsize=(5, 5))
plt.grid(color="grey",linestyle = '--', linewidth = 0.5)

plt.rcParams['font.family'] = 'Helvetica'
font = fm.FontProperties('Helvetica')

plt.grid(color="grey",linestyle = '--', linewidth = 0.1)

_shp_cities['coords'] = _shp_cities['geometry'].apply(lambda x: x.representative_point().coords[:])
_shp_cities['coords'] = [coords[0] for coords in _shp_cities['coords']]

_shp_cities["coords"].tolist()
_shp_cities[['lat', 'lon', 'zero']] = gpd.GeoDataFrame(_shp_cities['coords'].tolist(), index=_shp_cities.index)

x = _shp_cities["lat"]
y = _shp_cities["lon"]    
labels =_shp_cities["name"]

for i in range(0,len(_shp_cities)):
    plt.text(x[i]+0.008,y[i]+0.008,labels[i],fontsize=4,color = '#c3c3c3', ha = 'center')
    

dx = great_distance(start_latitude=11.1, start_longitude=-79.5, end_latitude=11.1, end_longitude=-79.6)
scalebar = ScaleBar(dx = 109250.50301657, location ="lower left", frameon=False, color='lightgrey', sep=1.5, width_fraction=0.012)
ax6.add_artist(scalebar)
scalebar.font_properties.set_size(5.5)
scalebar.font_properties.set_family('Helvetica')

ax6.tick_params(axis='x', colors='grey', labelsize=3, labeltop = 'True', labelrotation = 270)
ax6.tick_params(axis='y', colors='grey', labelsize=3, labelright = 'True') #reducing the size of the axis values

ax6.xaxis.set_major_formatter(FormatStrFormatter('%.2f')) #axis value formatting for both axis
ax6.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax6.set_xlim(79.40, 80)
ax6.set_ylim(10.93, 11.45)

ax6.spines['bottom'].set_color('none')
ax6.spines['top'].set_color('none') 
ax6.spines['right'].set_color('none')
ax6.spines['left'].set_color('none')

x, y, arrow_length = 0.5, 0.99, 0.1
ax6.annotate('N',color= "lightgrey", xy=(x, y), xytext=(x, y-arrow_length), arrowprops=dict(facecolor='none',edgecolor="lightgrey", width=4, headwidth=12), ha='center', va='center', fontsize=10, xycoords=ax6.transAxes)


_plt_district = _shp_district.plot(ax=ax6, figsize =(5,5),color="none",linewidth = 0.5)

_shp_cities.plot(ax=ax6, markersize=3, color='grey', zorder = 0)
#shp_land_cover_Barren.plot(color="#424242",ax =ax7, label ='Unused')
#shp_land_cover_sparseveg.plot(color="#f096ff",ax =ax7, label = 'Sparse vegetation')
#shp_land_cover_cropland.plot(color="#18a558",ax =ax7, label = 'Cropland', zorder = 0)
#shp_land_cover_Forest.plot(color="#B1D355",ax =ax7, label = 'Tree cover')
shp_land_cover_builtup.plot(color="#e73429",ax =ax6, label = 'Built-up', zorder=1)
#shp_land_cover_Water.plot(color="#00B9F2",ax =ax7, label = 'Water')
#shp_land_cover_Water1.plot(color="#00B9F2",ax =ax7)

#ax7.legend(loc='upper left', bbox_to_anchor=(0.8, 0.2), title = 'Legend\n', fontsize = 5.5, markerscale = 2, title_fontsize = 5.5, framealpha= 0, borderpad = 0.3, handletextpad = 0.5, handlelength = 1.0)

#Unused = mpatches.Patch(color="#424242", label ='Unused')
#SpV = mpatches.Patch(color="#f096ff", label = 'Sparse vegetation')
#Crop = mpatches.Patch(color="#18a558", label = 'Cropland')
#TC = mpatches.Patch(color="#B1D355", label = 'Tree cover')
BU = mpatches.Patch(color="#e73429", label = 'Built-up')
#Water = mpatches.Patch(color="#00B9F2", label = 'Water')

    
plt.legend(handles = [BU], loc = 'upper left', bbox_to_anchor=(0.8, 0.2), title = 'Legend\n', fontsize = 5.5, markerscale = 2, title_fontsize = 5.5, framealpha= 0, borderpad = 0.3, handletextpad = 0.5, handlelength = 1.0)


plt.savefig(get_in_workdir("Builtup.jpg"),dpi =1500)
plt.show()

# %% [markdown]
# ## Landcover as shapefile

# %%
shp_land_cover = read_df_UT('extrainputs/LC_mayil_shapefile.shp')
shp_land_cover["DN"].unique()

# %%
shp_land_cover_builtup = shp_land_cover[shp_land_cover["DN"] == 6 ]
shp_land_cover_Barren = shp_land_cover[shp_land_cover["DN"] == 1 ]
shp_land_cover_sparseveg = shp_land_cover[shp_land_cover["DN"] == 2 ]
shp_land_cover_cropland  = shp_land_cover[shp_land_cover["DN"] == 3 ]
shp_land_cover_Forest = shp_land_cover[shp_land_cover["DN"] == 4 ]
shp_land_cover_Water = shp_land_cover[shp_land_cover["DN"] == 5 ]
shp_land_cover_Water1 = shp_land_cover[shp_land_cover["DN"] == 50 ]

# %%
plt.rcParams['font.family'] = 'Helvetica'

_shp_cities = read_gpd_UT("extrainputs/Mayiladuthurai_major_towns.shp")

fig7, ax7 = plt.subplots(figsize=(5, 5))
plt.grid(color="grey",linestyle = '--', linewidth = 0.5)

plt.rcParams['font.family'] = 'Helvetica'
font = fm.FontProperties('Helvetica')

plt.grid(color="grey",linestyle = '--', linewidth = 0.1)

_shp_cities['coords'] = _shp_cities['geometry'].apply(lambda x: x.representative_point().coords[:])
_shp_cities['coords'] = [coords[0] for coords in _shp_cities['coords']]

_shp_cities["coords"].tolist()
_shp_cities[['lat', 'lon', 'zero']] = gpd.GeoDataFrame(_shp_cities['coords'].tolist(), index=_shp_cities.index)

x = _shp_cities["lat"]
y = _shp_cities["lon"]    
labels =_shp_cities["name"]

for i in range(0,len(_shp_cities)):
    plt.text(x[i]+0.008,y[i]+0.008,labels[i],fontsize=4,color = '#c3c3c3', ha = 'center')
    

dx = great_distance(start_latitude=11.1, start_longitude=-79.5, end_latitude=11.1, end_longitude=-79.6)
scalebar = ScaleBar(dx = 109250.50301657, location ="lower left", frameon=False, color='lightgrey', sep=1.5, width_fraction=0.012)
ax7.add_artist(scalebar)
scalebar.font_properties.set_size(5.5)
scalebar.font_properties.set_family('Helvetica')

ax7.tick_params(axis='x', colors='grey', labelsize=3, labeltop = 'True', labelrotation = 270)
ax7.tick_params(axis='y', colors='grey', labelsize=3, labelright = 'True') #reducing the size of the axis values

ax7.xaxis.set_major_formatter(FormatStrFormatter('%.2f')) #axis value formatting for both axis
ax7.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax7.set_xlim(79.40, 80)
ax7.set_ylim(10.93, 11.45)

ax7.spines['bottom'].set_color('none')
ax7.spines['top'].set_color('none') 
ax7.spines['right'].set_color('none')
ax7.spines['left'].set_color('none')

x, y, arrow_length = 0.5, 0.99, 0.1
ax7.annotate('N',color= "lightgrey", xy=(x, y), xytext=(x, y-arrow_length), arrowprops=dict(facecolor='none',edgecolor="lightgrey", width=4, headwidth=12), ha='center', va='center', fontsize=10, xycoords=ax7.transAxes)


_plt_district = _shp_district.plot(ax=ax7, figsize =(5,5),color="none",linewidth = 0.5)
#_shp_dst_roads_secondary.plot(color="#f0b8b3",label ="Secondary roads",ax=ax4, linewidth=0.5)
#_shp_dst_roads_primary.plot(color="#df2a29",label ="Primary roads",ax=ax4, linewidth=0.5)
#_shp_dst_railways.plot(color="#da0404",label ="Railway roads",ax=ax4, linestyle='--', linewidth=0.5)
_shp_cities.plot(ax=ax7, markersize=3, color='grey', zorder = 1)
shp_land_cover_Barren.plot(color="#424242",ax =ax7, label ='Unused')
shp_land_cover_sparseveg.plot(color="#f096ff",ax =ax7, label = 'Sparse vegetation')
shp_land_cover_cropland.plot(color="#18a558",ax =ax7, label = 'Cropland', zorder = 0)
shp_land_cover_Forest.plot(color="#B1D355",ax =ax7, label = 'Tree cover')
shp_land_cover_builtup.plot(color="#e73429",ax =ax7, label = 'Built-up')
shp_land_cover_Water.plot(color="#00B9F2",ax =ax7, label = 'Water')
shp_land_cover_Water1.plot(color="#00B9F2",ax =ax7)

#ax7.legend(loc='upper left', bbox_to_anchor=(0.8, 0.2), title = 'Legend\n', fontsize = 5.5, markerscale = 2, title_fontsize = 5.5, framealpha= 0, borderpad = 0.3, handletextpad = 0.5, handlelength = 1.0)

Unused = mpatches.Patch(color="#424242", label ='Unused')
SpV = mpatches.Patch(color="#f096ff", label = 'Sparse vegetation')
Crop = mpatches.Patch(color="#18a558", label = 'Cropland')
TC = mpatches.Patch(color="#B1D355", label = 'Tree cover')
BU = mpatches.Patch(color="#e73429", label = 'Built-up')
Water = mpatches.Patch(color="#00B9F2", label = 'Water')

    
plt.legend(handles = [Unused, SpV, Crop, TC, BU, Water], loc = 'upper left', bbox_to_anchor=(0.8, 0.2), title = 'Legend\n', fontsize = 5.5, markerscale = 2, title_fontsize = 5.5, framealpha= 0, borderpad = 0.3, handletextpad = 0.5, handlelength = 1.0)


plt.savefig(get_in_workdir("Landcover.jpg"),dpi =1500)
plt.show()

# %%

# %% [markdown]
# # Land suitability for solar

# %% [markdown]
# ### Technical Suitability - Technical, theoretical, and no potential lands

# %%
lc_tech = read_df_UT('solar/_rl_elev_rd_wat_co_trans_ar_sub_rdpx_trsub_tech/LC_Solar_final_area_mask_1_Nagapattinam.shp')

# %%
lc_theo = read_df_UT('solar/_rl_elev_rd_wat_co_th/LC_Solar_final_mask_val_1_Nagapattinam.shp')

# %%
lc_barren = read_df_UT('solar/all_lands_barren/all_BarrenLands_Mayu.shp')

# %%
#lc_tech.plot()
#lc_theo.plot()
#lc_barren.plot()

# %%
plt.rcParams['font.family'] = 'Helvetica'

_shp_cities = read_gpd_UT("extrainputs/Mayiladuthurai_major_towns.shp")

fig8, ax8 = plt.subplots(figsize=(5, 5))
plt.grid(color="grey",linestyle = '--', linewidth = 0.5)

plt.rcParams['font.family'] = 'Helvetica'
font = fm.FontProperties('Helvetica')

plt.grid(color="grey",linestyle = '--', linewidth = 0.1)

_shp_cities['coords'] = _shp_cities['geometry'].apply(lambda x: x.representative_point().coords[:])
_shp_cities['coords'] = [coords[0] for coords in _shp_cities['coords']]

_shp_cities["coords"].tolist()
_shp_cities[['lat', 'lon', 'zero']] = gpd.GeoDataFrame(_shp_cities['coords'].tolist(), index=_shp_cities.index)

x = _shp_cities["lat"]
y = _shp_cities["lon"]    
labels =_shp_cities["name"]

for i in range(0,len(_shp_cities)):
    ax8.text(x[i]+0.008,y[i]+0.008,labels[i],fontsize=4,color = '#c3c3c3', ha = 'center', zorder=0)

dx = great_distance(start_latitude=11.1, start_longitude=-79.5, end_latitude=11.1, end_longitude=-79.6)
scalebar = ScaleBar(dx = 109250.50301657, location ="lower left", frameon=False, color='lightgrey', sep=1.5, width_fraction=0.012)
ax8.add_artist(scalebar)
scalebar.font_properties.set_size(5.5)
scalebar.font_properties.set_family('Helvetica')

ax8.tick_params(axis='x', colors='grey', labelsize=3, labeltop = 'True', labelrotation = 270)
ax8.tick_params(axis='y', colors='grey', labelsize=3, labelright = 'True') #reducing the size of the axis values

ax8.xaxis.set_major_formatter(FormatStrFormatter('%.2f')) #axis value formatting for both axis
ax8.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax8.set_xlim(79.40, 80)
ax8.set_ylim(10.93, 11.45)

ax8.spines['bottom'].set_color('none')
ax8.spines['top'].set_color('none') 
ax8.spines['right'].set_color('none')
ax8.spines['left'].set_color('none')

x, y, arrow_length = 0.5, 0.99, 0.1
ax8.annotate('N',color= "lightgrey", xy=(x, y), xytext=(x, y-arrow_length), arrowprops=dict(facecolor='none',edgecolor="lightgrey", width=4, headwidth=12), ha='center', va='center', fontsize=10, xycoords=ax8.transAxes)


_shp_district.plot(figsize=(5,5),color="none", ax=ax8, linewidth = 0.5, zorder=5)
_shp_cities.plot(ax=ax8, markersize=3, color='grey', zorder=-1)
    

lc_barren.plot(color="#424242",ax =ax8, label='Low Potential')
lc_theo.plot(color="#997D41",ax =ax8, label='Medium Potential')
lc_tech.plot(color="#FBAF30",ax =ax8, label='High Potential')


No_P = mpatches.Patch(color='#424242', label='No potential')
Theo_P = mpatches.Patch(color='#997D41', label='Theoretical potential')
Tech_P = mpatches.Patch(color='#FBAF30', label='Technical potential')
    
plt.legend(handles = [No_P, Theo_P, Tech_P], loc = 'upper left', bbox_to_anchor=(0.8, 0.2), title = 'Legend\n', fontsize = 5.5, markerscale = 2, title_fontsize = 5.5, framealpha= 0, borderpad = 0.3, handletextpad = 0.5, handlelength = 1.0)


#ax10.legend(loc='upper left', bbox_to_anchor=(0.8, 0.2), title = 'Legend\n', fontsize = 5.5, markerscale = 2, title_fontsize = 5.5, framealpha= 0, borderpad = 0.3, handletextpad = 0.5, handlelength = 1.0)


print(plt.rcParams['font.family'])


plt.savefig(get_in_workdir("Technical_suitability.jpg"),dpi =1500)
plt.show()

# %% [markdown]
# ### Technical lands distributed by size

# %%
lc_tech.head()

# %%
lc_tech["area_class"].unique()

# %%
lc_tech_S3 = lc_tech[lc_tech["area_class"] == "A"]
lc_tech_S2 = lc_tech[lc_tech["area_class"] == "B"]
lc_tech_S1 = lc_tech[lc_tech["area_class"] == "C"]

# %%
plt.rcParams['font.family'] = 'Helvetica'

_shp_cities = read_gpd_UT("extrainputs/Mayiladuthurai_major_towns.shp")

fig9, ax9 = plt.subplots(figsize=(5, 5))
plt.grid(color="grey",linestyle = '--', linewidth = 0.5)

plt.rcParams['font.family'] = 'Helvetica'
font = fm.FontProperties('Helvetica')

plt.grid(color="grey",linestyle = '--', linewidth = 0.1)

_shp_cities['coords'] = _shp_cities['geometry'].apply(lambda x: x.representative_point().coords[:])
_shp_cities['coords'] = [coords[0] for coords in _shp_cities['coords']]

_shp_cities["coords"].tolist()
_shp_cities[['lat', 'lon', 'zero']] = gpd.GeoDataFrame(_shp_cities['coords'].tolist(), index=_shp_cities.index)

x = _shp_cities["lat"]
y = _shp_cities["lon"]    
labels =_shp_cities["name"]

for i in range(0,len(_shp_cities)):
    ax9.text(x[i]+0.008,y[i]+0.008,labels[i],fontsize=4,color = '#c3c3c3', ha = 'center', zorder=0)

dx = great_distance(start_latitude=11.1, start_longitude=-79.5, end_latitude=11.1, end_longitude=-79.6)
scalebar = ScaleBar(dx = 109250.50301657, location ="lower left", frameon=False, color='lightgrey', sep=1.5, width_fraction=0.012)
ax9.add_artist(scalebar)
scalebar.font_properties.set_size(5.5)
scalebar.font_properties.set_family('Helvetica')

ax9.tick_params(axis='x', colors='grey', labelsize=3, labeltop = 'True', labelrotation = 270)
ax9.tick_params(axis='y', colors='grey', labelsize=3, labelright = 'True') #reducing the size of the axis values

ax9.xaxis.set_major_formatter(FormatStrFormatter('%.2f')) #axis value formatting for both axis
ax9.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax9.set_xlim(79.40, 80)
ax9.set_ylim(10.93, 11.45)

ax9.spines['bottom'].set_color('none')
ax9.spines['top'].set_color('none') 
ax9.spines['right'].set_color('none')
ax9.spines['left'].set_color('none')

x, y, arrow_length = 0.5, 0.99, 0.1
ax9.annotate('N',color= "lightgrey", xy=(x, y), xytext=(x, y-arrow_length), arrowprops=dict(facecolor='none',edgecolor="lightgrey", width=4, headwidth=12), ha='center', va='center', fontsize=10, xycoords=ax9.transAxes)


_shp_district.plot(figsize=(5,5),color="none", ax=ax9, linewidth = 0.5, zorder=5)
_shp_cities.plot(ax=ax9, markersize=3, color='grey', zorder=-1)
    

lc_tech_S3.plot(color="#646464",ax =ax9, label='>5 to 20 acres')
lc_tech_S2.plot(color="#BDA383",ax =ax9, label='>20 to 100 acres')
lc_tech_S1.plot(color="#FBAF30",ax =ax9, label='>100 acres')


S1 = mpatches.Patch(color='#646464', label='>5 to 20 acres')
S2 = mpatches.Patch(color='#BDA383', label='>20 to 100 acres')
S3 = mpatches.Patch(color='#FBAF30', label='>100 acres')
    
plt.legend(handles = [S1, S2, S3], loc = 'upper left', bbox_to_anchor=(0.8, 0.2), title = 'Legend\n', fontsize = 5.5, markerscale = 2, title_fontsize = 5.5, framealpha= 0, borderpad = 0.3, handletextpad = 0.5, handlelength = 1.0)


#ax10.legend(loc='upper left', bbox_to_anchor=(0.8, 0.2), title = 'Legend\n', fontsize = 5.5, markerscale = 2, title_fontsize = 5.5, framealpha= 0, borderpad = 0.3, handletextpad = 0.5, handlelength = 1.0)


print(plt.rcParams['font.family'])


plt.savefig(get_in_workdir("Distribution by size.jpg"),dpi =1500)
plt.show()

# %%

# %% [markdown]
# ### High Potential - High, medium, low potential lands

# %%
lc_high = read_df_UT('solar/_rl_elev_rd_wat_co_trans_ar_sub_rdpx_trsub_trat_subat_rdat_ir_high/LC_Solar_final_area_mask_1_Nagapattinam.shp')

# %%
lc_med = read_df_UT("solar/_rl_elev_rd_wat_co_trans_ar_sub_rdpx_trsub_trat_subat_rdat_ir_medatt/LC_Solar_final_area_mask_1_Nagapattinam.shp")

# %%
lc_low = read_df_UT('solar/_rl_elev_rd_wat_trans_ar_sub_rdpx_trsub_low/LC_Solar_final_area_mask_1_Nagapattinam.shp')

# %%
merge_lc_high = overlay(lc_high,_shp_district,how ="intersection")
merge_lc_med = overlay(lc_med,_shp_district,how ="intersection")
merge_lc_low = overlay(lc_low,_shp_district,how ="intersection")

# %%
#merge_lc_low.head()

# %%
plt.rcParams['font.family'] = 'Helvetica'

_shp_cities = read_gpd_UT("extrainputs/Mayiladuthurai_major_towns.shp")

fig10, ax10 = plt.subplots(figsize=(5, 5))
plt.grid(color="grey",linestyle = '--', linewidth = 0.5)

plt.rcParams['font.family'] = 'Helvetica'
font = fm.FontProperties('Helvetica')

plt.grid(color="grey",linestyle = '--', linewidth = 0.1)

_shp_cities['coords'] = _shp_cities['geometry'].apply(lambda x: x.representative_point().coords[:])
_shp_cities['coords'] = [coords[0] for coords in _shp_cities['coords']]

_shp_cities["coords"].tolist()
_shp_cities[['lat', 'lon', 'zero']] = gpd.GeoDataFrame(_shp_cities['coords'].tolist(), index=_shp_cities.index)

x = _shp_cities["lat"]
y = _shp_cities["lon"]    
labels =_shp_cities["name"]

for i in range(0,len(_shp_cities)):
    ax10.text(x[i]+0.008,y[i]+0.008,labels[i],fontsize=4,color = '#c3c3c3', ha = 'center', zorder=0)

dx = great_distance(start_latitude=11.1, start_longitude=-79.5, end_latitude=11.1, end_longitude=-79.6)
scalebar = ScaleBar(dx = 109250.50301657, location ="lower left", frameon=False, color='lightgrey', sep=1.5, width_fraction=0.012)
ax10.add_artist(scalebar)
scalebar.font_properties.set_size(5.5)
scalebar.font_properties.set_family('Helvetica')

ax10.tick_params(axis='x', colors='grey', labelsize=3, labeltop = 'True', labelrotation = 270)
ax10.tick_params(axis='y', colors='grey', labelsize=3, labelright = 'True') #reducing the size of the axis values

ax10.xaxis.set_major_formatter(FormatStrFormatter('%.2f')) #axis value formatting for both axis
ax10.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax10.set_xlim(79.40, 80)
ax10.set_ylim(10.93, 11.45)

ax10.spines['bottom'].set_color('none')
ax10.spines['top'].set_color('none') 
ax10.spines['right'].set_color('none')
ax10.spines['left'].set_color('none')

x, y, arrow_length = 0.5, 0.99, 0.1
ax10.annotate('N',color= "lightgrey", xy=(x, y), xytext=(x, y-arrow_length), arrowprops=dict(facecolor='none',edgecolor="lightgrey", width=4, headwidth=12), ha='center', va='center', fontsize=10, xycoords=ax10.transAxes)


_shp_district.plot(figsize=(5,5),color="none", ax=ax10, linewidth = 0.5, zorder=5)
_shp_cities.plot(ax=ax10, markersize=3, color='grey', zorder=-1)
    

merge_lc_low.plot(color="#5A3228",ax =ax10, label='Low Potential')
merge_lc_med.plot(color="#A77145",ax =ax10, label='Medium Potential')
merge_lc_high.plot(color="#FBAF31",ax =ax10, label='High Potential')


Low_P = mpatches.Patch(color='#5A3228', label='Low potential')
Med_P = mpatches.Patch(color='#A77145', label='Medium potential')
High_P = mpatches.Patch(color='#FBAF31', label='High potential')
    
plt.legend(handles = [Low_P, Med_P, High_P], loc = 'upper left', bbox_to_anchor=(0.8, 0.2), title = 'Legend\n', fontsize = 5.5, markerscale = 2, title_fontsize = 5.5, framealpha= 0, borderpad = 0.3, handletextpad = 0.5, handlelength = 1.0)


#ax10.legend(loc='upper left', bbox_to_anchor=(0.8, 0.2), title = 'Legend\n', fontsize = 5.5, markerscale = 2, title_fontsize = 5.5, framealpha= 0, borderpad = 0.3, handletextpad = 0.5, handlelength = 1.0)


print(plt.rcParams['font.family'])


plt.savefig(get_in_workdir("High Potential_H_M_L.jpg"),dpi =1500)
plt.show()

# %% [markdown]
# ## Competing Land Use

# %%
solar_tech = read_df_UT('solar/_rl_elev_rd_wat_co_trans_ar_sub_rdpx_trsub_tech/LC_Solar_final_area_mask_1_Nagapattinam.shp')

# %%
#solar_tech.head(), solar_tech['area_class'].unique()

# %%
solar_tech_A = solar_tech[solar_tech["area_class"] == "A"]
solar_tech_B = solar_tech[solar_tech["area_class"] == "B"]
solar_tech_C = solar_tech[solar_tech["area_class"] == "C"]

# %%
tech_ABC = gpd.pd.concat([solar_tech_A,solar_tech_B,solar_tech_C])
#tech_ABC.head(), tech_ABC['area_class'].unique()

# %%
_shp_water_high =  read_df_UT('water/_wd_run_high/LC_Water_final.shp')
_shp_water_med = read_df_UT('water/_wd_run_med/LC_Water_final.shp')

_shp_water_high.shape, _shp_water_med.shape

# %% [markdown]
# Solar Tech + Water High and Medium

# %%
water_high_dist = overlay(_shp_water_high,_shp_district,how ="intersection")
water_med_dist = overlay(_shp_water_med,_shp_district,how ="intersection")

water_high_dist.shape, water_med_dist.shape

# %%
water_high_dist.head()

# %%
water_high_med_dist = gpd.pd.concat([water_high_dist, water_med_dist])

# %%
_shp_merge_tech_water = overlay(water_high_med_dist,tech_ABC, how='intersection')
#_shp_merge_tech_water.plot()
#water_high_med_dist.plot()
#tech_ABC.plot()

# %% [markdown]
# Solar Tech + Forest High and Medium

# %%
forest_med = read_df_UT('forest/_ter_elev_watpot_ar_med/LC_Forest_final_area_mask_1_Nagapattinam.shp')
forest_med.plot()

# %%
_shp_merge_tech_forest = overlay(forest_med, tech_ABC, how='intersection')

# %%
#_shp_merge_tech_forest.columns
#_shp_merge_tech_forest["geometry"].plot()

# %% [markdown]
# Solar Tech + Forest High and Medium + Water High and Medium

# %%
_shp_merge_tech_water_forest = overlay(_shp_merge_tech_water,_shp_merge_tech_forest, how='intersection')

# %%
#_shp_merge_tech_water_forest["geometry"].plot()

# %% [markdown]
# Competing land use plot

# %%
plt.rcParams['font.family'] = 'Helvetica'

_shp_cities = read_gpd_UT("extrainputs/Mayiladuthurai_major_towns.shp")

fig11, ax11 = plt.subplots(figsize=(5, 5))
plt.grid(color="grey",linestyle = '--', linewidth = 0.5)

plt.rcParams['font.family'] = 'Helvetica'
font = fm.FontProperties('Helvetica')

plt.grid(color="grey",linestyle = '--', linewidth = 0.1)

_shp_cities['coords'] = _shp_cities['geometry'].apply(lambda x: x.representative_point().coords[:])
_shp_cities['coords'] = [coords[0] for coords in _shp_cities['coords']]

_shp_cities["coords"].tolist()
_shp_cities[['lat', 'lon', 'zero']] = gpd.GeoDataFrame(_shp_cities['coords'].tolist(), index=_shp_cities.index)

x = _shp_cities["lat"]
y = _shp_cities["lon"]    
labels =_shp_cities["name"]

for i in range(0,len(_shp_cities)):
    ax11.text(x[i]+0.008,y[i]+0.008,labels[i],fontsize=4,color = '#c3c3c3', ha = 'center', zorder=0)

dx = great_distance(start_latitude=11.1, start_longitude=-79.5, end_latitude=11.1, end_longitude=-79.6)
scalebar = ScaleBar(dx = 109250.50301657, location ="lower left", frameon=False, color='lightgrey', sep=1.5, width_fraction=0.012)
ax11.add_artist(scalebar)
scalebar.font_properties.set_size(5.5)
scalebar.font_properties.set_family('Helvetica')

ax11.tick_params(axis='x', colors='grey', labelsize=3, labeltop = 'True', labelrotation = 270)
ax11.tick_params(axis='y', colors='grey', labelsize=3, labelright = 'True') #reducing the size of the axis values

ax11.xaxis.set_major_formatter(FormatStrFormatter('%.2f')) #axis value formatting for both axis
ax11.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax11.set_xlim(79.40, 80)
ax11.set_ylim(10.93, 11.45)

ax11.spines['bottom'].set_color('none')
ax11.spines['top'].set_color('none') 
ax11.spines['right'].set_color('none')
ax11.spines['left'].set_color('none')

x, y, arrow_length = 0.5, 0.99, 0.1
ax11.annotate('N',color= "lightgrey", xy=(x, y), xytext=(x, y-arrow_length), arrowprops=dict(facecolor='none',edgecolor="lightgrey", width=4, headwidth=12), ha='center', va='center', fontsize=10, xycoords=ax11.transAxes)


_shp_district.plot(figsize=(5,5),color="none", ax=ax11, linewidth = 0.5, zorder=5)
_shp_cities.plot(ax=ax11, markersize=3, color='grey', zorder=-1)
    

tech_ABC.plot(color="#FBB034",ax =ax11)
_shp_merge_tech_water.plot(color="#00B9F2",ax =ax11)
_shp_merge_tech_forest.plot(color="#B1D355",ax =ax11)
#_shp_merge_tech_water_forest.plot(color="#e73429",ax =ax11, label='>100 acres')


S = mpatches.Patch(color='#FBB034', label='Solar only')
F = mpatches.Patch(color='#B1D355', label='Solar & forest')
W = mpatches.Patch(color='#00B9F2', label='Solar & water')
FnW = mpatches.Patch(color='#e73429', label='Solar, forest & water')
    
plt.legend(handles = [S, F, W, FnW], loc = 'upper left', bbox_to_anchor=(0.8, 0.2), title = 'Legend\n', fontsize = 5.5, markerscale = 2, title_fontsize = 5.5, framealpha= 0, borderpad = 0.3, handletextpad = 0.5, handlelength = 1.0)


#ax10.legend(loc='upper left', bbox_to_anchor=(0.8, 0.2), title = 'Legend\n', fontsize = 5.5, markerscale = 2, title_fontsize = 5.5, framealpha= 0, borderpad = 0.3, handletextpad = 0.5, handlelength = 1.0)


print(plt.rcParams['font.family'])


plt.savefig(get_in_workdir("Competing Land Use.jpg"),dpi =1500)
plt.show()

# %% [markdown]
# ## Landcover

# %%
#read landcover array
lc = read_raster_UT('LC_Mayiladhuthurai_epsg32644.tif')
lc_array1 = lc.read()

lc_array1, lc_array1.shape

# %%
#Read all values in a discrete raster - method 1
values = np.unique(lc_array1)
values

# %% [markdown]
# List of values for each landcover type:
#
# -9999 - nan
#
# 0 - invalid pixels
#
# 1 - barren
#
# 2 - sparse veg/scrubs
#
# 3 - crop land 
#
# 4 - Forest
#
# 5 - Water - delta
#
# 6 - Urban
#
# 50 - Water - lakes and rivers

# %% [markdown]
# Something better than nothing 1

# %% [markdown]
# Method 3

# %%
#Read the meta data with rasterio
lc_builtup_transform = lc.transform

lc_builtup_crs = lc.crs

lc_builtup_transform, lc_builtup_crs, lc.meta
lc_array1.shape

# %%
rasterio.plot.show(lc)

# %%
#Make copy of array
lc_builtup = lc_array1.copy()

# %%
#Pick a particular layer and set rest of the values as -9999

lc_builtup[lc_builtup == 6] = 6
lc_builtup[lc_builtup != 6] = -9999
lc_builtup

# %%
values = np.unique(lc_builtup)
values

# %%
lc_builtup.dtype

# %%
#Convert array into integer
lc_builtup[np.isnan(lc_builtup)] = -9999
lc_builtup_int = lc_builtup.astype(np.int16)
lc_builtup_int

# %%
lc_builtup_int.dtype, lc_builtup_int.shape, lc_array1.shape, np.unique(lc_builtup_int)

# %%
built_up_dataset = rasterio.open(get_rooted("builtup6.tif"), "w", 
    driver = "GTiff",
    height = lc_builtup_int.shape[0],
    width = lc_builtup_int.shape[1],
    count = 1,
    nodata = -9999,
    dtype = 'int16',
    crs = lc.crs,
    transform = lc.transform
)
built_up_dataset.write(lc_builtup_int)
built_up_dataset.close()

# %%
image = rasterio.open(get_rooted("builtup6.tif"))
image.read()

# %%
#rasterio.plot.show(image.read(), cmap='gray_r', transform=image.transform)

# %%
#image.meta

# %%

# %%

# %%

# %%

# %%

# %%

# %%
