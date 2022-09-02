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
import rasterio.mask
import fiona
import rasterio
import geopandas as gpd
from osgeo import ogr, gdal
from osgeo import gdal_array
from osgeo import gdalconst


# %%
#Setting path to directory

def get_rooted(stem):
    return "F:/AV Consulting/2022/LiLa/TN Government/Data/LiLa_Nagapattinam/" + stem

def read_raster_UT(stem):
    return rasterio.open(get_rooted(stem))

def read_shpfi_UT(stem):
    return fiona.open(get_rooted(stem))    

def read_rastergpd_UT(stem):
    return gpd.read_file(get_rooted(stem)) 

def read_rastergdal_UT(stem):
    return gdal.Open(get_rooted(stem))    

    

# %%
#Open Shapefile with fiona

# %%
with read_shpfi_UT('Practice/Nagapattinam_proj32644.shp') as shapefile:
    for feature in shapefile:
        shapes = [feature['geometry']]



# %%
#Coordinate Reference System of Shapefile

read_shpfi_UT('Practice/Nagapattinam_proj32644.shp').crs

# %% tags=[]
#Open Raster #with rasterio

# %%
with read_raster_UT('Supporting _info/DEM_T44PLT_proj32644_filled_slope.tif') as src:
    out_image, out_transform = rasterio.mask.mask(src, shapes, crop = True)
    out_meta = src.meta

# %%
#Coordinate Reference System of Shapefile

read_raster_UT('Supporting _info/DEM_T44PLT_proj32644_filled_slope.tif').crs

# %%
out_meta.update({
    'driver':'Gtiff',
    'height': out_image.shape[1],
    'width': out_image.shape[2],
    'transform': out_transform,
})    

# %%
out_meta = src.meta

# %%
with rasterio.open('F:/AV Consulting/2022/LiLa/TN Government/Data/LiLa_Nagapattinam/workdir/Outputslope_rasternagapattinam_slope.tif', 'w', **out_meta) as dst:
    dst.write(out_image)

# %%
import rasterio.plot

# %%
rasterio.plot.show(read_raster_UT('workdir/Outputslope_rasternagapattinam_slope.tif'))

# %% [markdown]
# Converting raster into epsg 4326 CRS

# %%
#Open raster
_tif_dst_slope = read_raster_UT('workdir/Outputslope_rasternagapattinam_slope.tif')

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
_tif_dst_slope_epsg4326 = rasterio.open(
    'F:/AV Consulting/2022/LiLa/TN Government/Data/LiLa_Nagapattinam/workdir/Outputslope_rst_dst_epsg4326.tif', 'w', **kwargs)


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
from rasterio.plot import show

# %%
_tif_dst_slope_epsg4326 = read_raster_UT('workdir/Outputslope_rst_dst_epsg4326.tif')
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

# %% [markdown]
# ## Slope Raster
#
# To visualize raster, information was used from:
#
# https://zia207.github.io/geospatial-python.io/lesson_06_working-with-raster-data.html
#
# https://stackoverflow.com/questions/61327088/rio-plot-show-with-colorbar
#
# https://matplotlib.org/stable/api/axis_api.html
#

# %%
gdal_data = read_rastergdal_UT('workdir/Outputslope_rst_dst_epsg4326.tif')
gdal_band = gdal_data.GetRasterBand(1)
nodataval = gdal_band.GetNoDataValue()

# convert to a numpy array
data_array = gdal_data.ReadAsArray().astype(np.float)
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
AVC_color = LinearSegmentedColormap.from_list('testCmap1', colors=['#ec6669','#FFFFFF'], N=256)


# %%
#Show min and max of array without nan values
a = np.nanmax(data_array)
b = np.nanmin(data_array)
a, b

# %%
import matplotlib.ticker as ticker
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# %%
dem = read_raster_UT('workdir/Outputslope_rst_dst_epsg4326.tif')

fig, ax = plt.subplots(figsize=(5, 5))
plt.grid(color="grey",linestyle = '--', linewidth = 0.5)


ax.xaxis.tick_top()

ax.tick_params(axis='x', colors='grey', labelsize=5)
ax.tick_params(axis='y', colors='grey', labelsize=5) #reducing the size of the axis values

ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f')) #axis value formatting for both axis
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax.spines['bottom'].set_color('none')
ax.spines['top'].set_color('none') 
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('none')

# use imshow so that we have something to map the colorbar to
image_hidden = ax.imshow(data_array, 
                         cmap=AVC_color, 
                         )

# plot on the same axis with rasterio.plot.show
image = rasterio.plot.show(dem, 
                      transform=dem.transform, 
                      ax=ax, 
                      cmap=AVC_color, 
                      )

# add colorbar using the now hidden image
fig.colorbar(image_hidden, ax=ax)

# %% [markdown]
# ## GHI

# %%
GHI_gdal_data = read_rastergdal_UT('workdir/GHIepsg_Nagapattinam.tif')
gdal_band = GHI_gdal_data.GetRasterBand(1)
nodataval = gdal_band.GetNoDataValue()

# convert to a numpy array
GHI_data_array = GHI_gdal_data.ReadAsArray().astype(np.float)
GHI_data_array

# replace missing values if necessary
if np.any(data_array == nodataval):
    GHI_data_array[data_array == nodataval] = np.nan
    GHI_data_array[data_array == nodataval] = np.nan

# %%
#Set colour
AVC_color2 = LinearSegmentedColormap.from_list('testCmap', colors=['#e62314','#f19e18'], N=256)

# %%
#Show min and max of array without nan values
a = np.nanmax(GHI_data_array)
b = np.nanmin(GHI_data_array)
a, b

# %%
GHI = read_raster_UT('workdir/GHIepsg_Nagapattinam.tif')

fig, ax = plt.subplots(figsize=(5, 5))
plt.grid(color="grey",linestyle = '--', linewidth = 0.5)


ax.xaxis.tick_top()

ax.tick_params(axis='x', colors='grey', labelsize=5)
ax.tick_params(axis='y', colors='grey', labelsize=5) #reducing the size of the axis values

ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f')) #axis value formatting for both axis
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax.spines['bottom'].set_color('none')
ax.spines['top'].set_color('none') 
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('none')

# use imshow so that we have something to map the colorbar to
image_hidden = ax.imshow(GHI_data_array, 
                         cmap=AVC_color2, 
                         vmin = 1970.524048,
                         vmax = a,
                         )

# plot on the same axis with rasterio.plot.show
image = rasterio.plot.show(GHI, 
                      transform=dem.transform, 
                      ax=ax, 
                      cmap=AVC_color2, 
                      with_bounds = 'True'
                      )

# add colorbar using the now hidden image
fig.colorbar(image_hidden, ax=ax)

# %%
show(GHI, cmap=AVC_color2)

# %%
fig, ax = plt.subplots(figsize=(5, 5))

image = rasterio.plot.show(GHI, 
                      transform=dem.transform, 
                      ax=ax, 
                      cmap=AVC_color2, 
                      with_bounds = 'True'
                      )

plt.grid(color="grey",linestyle = '--', linewidth = 0.5)


ax.xaxis.tick_top()

ax.tick_params(axis='x', colors='grey', labelsize=5)
ax.tick_params(axis='y', colors='grey', labelsize=5) #reducing the size of the axis values

ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f')) #axis value formatting for both axis
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax.spines['bottom'].set_color('none')
ax.spines['top'].set_color('none') 
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('none')


