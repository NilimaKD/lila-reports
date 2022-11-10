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
import pandas as pd
from matplotlib import pyplot
import matplotlib.pyplot as plt

from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.font_manager as fm
from pygc import great_distance
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.ticker import LinearLocator, FormatStrFormatter



# %%
#Setting path to directory

def get_rooted(stem):
    return "F:/AV Consulting/2022/LiLa/Hasten/Data/Panna/forest/" + stem

def get_rooted_water(stem):
    return "F:/AV Consulting/2022/LiLa/Hasten/Data/Panna/water/" + stem

def get_in_workdir(stem):
    if _is_kaggle or _is_colab:
        return './' + stem
    else:
        return get_rooted('workdir/' + stem)
    
def get_in_hyperlink(stem):
    if _is_kaggle or _is_colab:
        return './' + stem
    else:
        return get_rooted('Panna_hyperlink/' + stem)
    
def get_in_HV(stem):
    if _is_kaggle or _is_colab:
        return './' + stem
    else:
        return get_rooted('HastenVentures/' + stem)       

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

def read_df_UT32644(stem):
    return gpd.read_file(get_rooted(stem)).to_crs(epsg = 32644)
    
def read_df_water_UT(stem):
    return gpd.read_file(get_rooted_water(stem)).to_crs(epsg = 4326)


# %% [markdown]
# # Common Features

# %% [markdown]
# #### Method to create a shapefile of major towns/cities
#
# Create a KML file with the locations:
#
# For instructions: https://github.com/CenterForSpatialResearch/gis_tutorials/blob/master/19_Importing_and_Exporting_GIS_Data_from_Google_Earth_and_Google_Maps.md
# Create file here: https://www.google.com/maps/d/
#     
# Convert KML to shapefile here:
# https://mygeodata.cloud/converter/kml-to-shp

# %%
#Testing Major cities and towns shapefile
shp_cities = read_gpd_UT("extrainputs/Panna_major_towns_cities-point.shp")
shp_cities.head()

# %% [markdown]
# #### Method to create scalebar
#
# Check: https://pypi.org/project/matplotlib-scalebar/
#
# To create the scalebar it is necessary to calculate the distance between two points defined by latitude & longitude.
# For this we use Great Distance.
# Check: https://pypi.org/project/pygc/
#
# Note: For calculating the distance between two points the latitude needs to be the same and the difference in longitude needs to be 1.
#
# Can also check the calculation here: https://www.calculator.net/distance-calculator.html

# %%
#Calculating the distance between two points with longitude and latitude
dx = great_distance(start_latitude=24.4, start_longitude=-79, end_latitude=24.4, end_longitude=-80) 
dx


# %% [markdown]
# #### Defining functions for common features

# %%
def plot_cities(fig, ax):
    shp_cities = read_gpd_UT("extrainputs/Panna_major_towns_cities-point.shp")
    shp_cities['coords'] = shp_cities['geometry'].apply(lambda x: x.representative_point().coords[:])
    shp_cities['coords'] = [coords[0] for coords in shp_cities['coords']]
    shp_cities["coords"].tolist()
    shp_cities[['lat', 'lon']] = gpd.GeoDataFrame(shp_cities['coords'].tolist(), index=shp_cities.index)

    x = shp_cities["lat"]
    y = shp_cities["lon"]    
    labels =shp_cities["Name"]

    for i in range(0,len(shp_cities)):
        plt.text(x[i]+0.008,y[i]+0.008,labels[i],fontsize=4,color = '#c3c3c3', ha = 'center')

    shp_cities.plot(ax=ax, markersize=3, color='grey')

def plot_common_features(fig, ax):
    plt.rcParams['font.family'] = 'Helvetica'
    plt.grid(color="grey",linestyle = '--', linewidth = 0.1)

    scalebar = ScaleBar(dx = 101434, location = 'lower left', frameon=False, color='lightgrey', sep=1.5, width_fraction=0.012) 
    ax.add_artist(scalebar) 
    scalebar.font_properties.set_size(5.5)
    scalebar.font_properties.set_family('Helvetica')

    ax.tick_params(axis='x', colors='grey', labelsize=3, labeltop = 'True', labelrotation = 270)
    ax.tick_params(axis='y', colors='grey', labelsize=3, labelright = 'True') #reducing the size of the axis values
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f')) #axis value formatting for both axis
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xlim(79.44, 81.05)
    ax.set_ylim(23.61, 25.21)

    ax.spines['bottom'].set_color('none')
    ax.spines['top'].set_color('none') 
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')

    x, y, arrow_length = 0.5, 0.99, 0.1
    ax.annotate('N',color= "lightgrey", xy=(x, y), xytext=(x, y-arrow_length), arrowprops=dict(facecolor='none',edgecolor="lightgrey", width=4, headwidth=12), ha='center', va='center', fontsize=10, xycoords=ax.transAxes)


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
_shp_dst = read_gpd_UT('extrainputs/Panna_dst_32644.shp')
raster = read_raster_UT('DEM_T44_M_P_proj32644_filled_slope.tif')
fig, ax = plt.subplots(figsize=(5, 5))
plt.grid(color="grey",linestyle = '--', linewidth = 0.5)
image = rasterio.plot.show(raster, 
                      transform=raster.transform, 
                      ax=ax, 
                      cmap='gray_r', 
                      )
_shp_dst.plot(ax=ax, color='blue', alpha =0.3)

# %%
#Open Shapefile with fiona

# %%
_shp_dst.head()

# %%
with read_shpfi_UT('extrainputs/Panna_dst_32644.shp') as shapefile:
    for feature in shapefile:
        shapes = [feature['geometry']]


# %%
#Coordinate Reference System of Shapefile

read_shpfi_UT('extrainputs/Panna_dst_32644.shp').crs

# %%
#Coordinate Reference System of Rasterfile

read_raster_UT('DEM_T44_M_P_proj32644_filled_slope.tif').crs

# %%
#Meta data of raster
read_raster_UT('DEM_T44_M_P_proj32644_filled_slope.tif').meta

# %% tags=[]
#Open Raster with rasterio and mask raster with shapefile

# %%
with read_raster_UT('DEM_T44_M_P_proj32644_filled_slope.tif') as src:
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
with rasterio.open(get_in_workdir('slope_panna_dst.tif'), 'w', **out_meta) as dst:
    dst.write(out_image)

# %%
rasterio.plot.show(read_raster_UT('workdir/slope_panna_dst.tif'))

# %% [markdown]
# ##### Converting raster into epsg 4326 CRS

# %%
#Open raster
_tif_dst_slope = rasterio.open(get_in_workdir('slope_panna_dst.tif'))

# %%
_tif_dst_slope.meta

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
_tif_dst_slope_epsg4326 = rasterio.open(get_in_workdir('slope_panna_dst_epsg4326.tif'), 'w', **kwargs)


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

# %% [markdown]
# Note: Reprojecting a raster will invariable change the array, which means that the raster needs to be resampled - add values in new neighbouring cells. Resampling can be done in different ways - average, nearest, bilinear etc...
#
# Check:
# 1. https://rasterio.readthedocs.io/en/latest/topics/resampling.html
#
# 2. https://gis.stackexchange.com/questions/220487/is-it-normal-that-the-values-of-a-raster-will-change-after-reprojecting-it
#
# The appropriate *resampling* method depends on the type of raster.
# It is recommended to use nearest, bilinear or cubic when rasters are continuous. (The default resampling on QGIS is nearest).
# Check: https://support.esri.com/en/technical-article/000005606

# %%
_tif_dst_slope_epsg4326 = read_raster_UT('workdir/slope_panna_dst_epsg4326.tif')
show(_tif_dst_slope_epsg4326)

# %%
array = _tif_dst_slope_epsg4326.read()
array1 = _tif_dst_slope.read()

array.shape, array1.shape

#raster shape is given as (band, rows, columns)

# %%
_tif_dst_slope_epsg4326.colorinterp[0]

# %%
_tif_dst_slope_epsg4326_meta =_tif_dst_slope_epsg4326.meta
print(_tif_dst_slope_epsg4326_meta)

# %%
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
from osgeo import ogr, gdal
from osgeo import gdal_array
from osgeo import gdalconst

# %%
gdal_data = read_rastergdal_UT('workdir/slope_panna_dst_epsg4326.tif')
gdal_band = gdal_data.GetRasterBand(1)
nodataval = gdal_band.GetNoDataValue()

# convert to a numpy array
data_array = gdal_data.ReadAsArray().astype(float)
data_array

# replace missing values if necessary
if np.any(data_array == nodataval):
    data_array[data_array == nodataval] = np.nan
    data_array[data_array == nodataval] = np.nan

# %%
#Show min and max of array without nan values
a = np.nanmax(data_array)
b = np.nanmin(data_array)
a, b

# %%
slope = read_raster_UT('DEM_T44_M_P_proj32644_filled_slope.tif')

array_origin = slope.read()

show(array_origin), np.max(array_origin), np.min(array_origin), show(array1), np.max(array1), np.min(array1), show(array), np.max(array), np.min(array),

# %% [markdown]
# #### Generate image of slope features of district

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
slope = read_raster_UT('workdir/slope_panna_dst_epsg4326.tif')


fig1, ax1 = plt.subplots(figsize=(5, 5))

plot_common_features(fig1, ax1)
plot_cities(fig1, ax1)

# use imshow so that we have something to map the colorbar to
image_hidden = ax.imshow(data_array, 
                         cmap=AVC_grey, 
                         )


# add colorbar using the now hidden image
cbaxes = fig1.add_axes([0.78, 0.18, 0.05, 0.05], frameon=False, title = 'Legend\n \n Height\n above sea-level (m)') 

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

# %% [markdown]
# # Shapefiles

# %% [markdown]
# ## Forest Reserve

# %%
_shp_district = read_df_UT('extrainputs/panna_dst.shp')
_shp_forest_reserve = read_df_UT('protected_areas.geojson')
_shp_forest_reserve.plot(), _shp_district.plot()

# %%
_shp_dst_forest_reserve = overlay(_shp_forest_reserve, _shp_district, how='intersection')
_shp_dst_forest_reserve.plot()

# %%
fig2, ax2 = plt.subplots(figsize=(5, 5))

plot_common_features(fig2, ax2)
plot_cities(fig2, ax2)    

_plt_district = _shp_district.plot(ax=ax2, figsize =(5,5),color="none",linewidth = 0.5)
_shp_dst_forest_reserve.plot(color="#048160",ax =ax2, label = 'Forest reserve', zorder=3)

FR = mpatches.Patch(color="#048160", label = 'Forest reserve')
plt.legend(handles = [FR], loc = 'upper left', bbox_to_anchor=(0.8, 0.2), title = 'Legend\n', fontsize = 5.5, markerscale = 2, title_fontsize = 5.5, framealpha= 0, borderpad = 0.3, handletextpad = 0.5, handlelength = 1.0)


plt.savefig(get_in_workdir("Forest reserve.jpg"),dpi =1500)
plt.show()

# %% [markdown]
# ## Water Bodies 
# Rivers and Lakes

# %%
_shp_water_setcsr = read_gpd_UT("LC_w_urban_w_waterbod_proj32644_mask_val_5.shp").set_crs(epsg = 32644)
_shp_water = _shp_water_setcsr.to_crs(epsg = 4326)
_shp_district = read_df_UT('extrainputs/panna_dst.shp')


_shp_district.plot(), _shp_water.plot()

# %%
_shp_dst_water = overlay(_shp_water,_shp_district,how ="intersection")
_shp_dst_water.plot(color = 'blue')

# %% [markdown]
# This plot shows the water bodies within the boundaries of the district.
# Details of the library can be found here : https://geopandas.org/en/stable/docs/user_guide/mapping.html

# %%
#_shp_dst_water["color"] = "Waterbodies"

#_shp_dst_water["GEOMORPHOL"].unique()

# %%
plt.rcParams['font.family'] = 'Helvetica'

fig3, ax3 = plt.subplots(figsize=(5, 5))

plot_common_features(fig3, ax3)
plot_cities(fig3, ax3)

    
_shp_dst_waterbodies = _shp_dst_water.plot(figsize =(5,5), color='#00B9F2', categorical=True, ax=ax3, zorder = 3)
_plt_district = _shp_district.plot(ax=ax3, figsize =(5,5),color="none",linewidth = 0.4)

blue_patch = mpatches.Patch(color='#00B9F2', label='Waterbodies')
    
plt.legend(handles = [blue_patch], loc = 'upper left', bbox_to_anchor=(0.8, 0.2), title = 'Legend\n', fontsize = 5.5, markerscale = 2, title_fontsize = 5.5, framealpha= 0, borderpad = 0.3, handletextpad = 0.5, handlelength = 1.0)


print(plt.rcParams['font.family'])
plt.savefig(get_in_workdir("Waterbodies.jpg"),dpi =1500)
plt.show()

# %% [markdown]
# ## Accessibility

# %% [markdown]
# ### Primary and Secondary Roads

# %%
_shp_roads = read_df_UT("output_osmroad_edges_25.100577511_23.77347712394135_80.63891837090935_79.76759704000001.geojson/edges.shp")

_shp_roads.plot()

# %%
_shp_roads.shape , _shp_roads["highway"].unique()

# %%
_shp_roads_secondary = _shp_roads.apply(lambda row: row[_shp_roads['highway'].isin(['secondary'])])
_shp_roads_primary = _shp_roads.apply(lambda row: row[_shp_roads['highway'].isin(['primary'])])
_shp_roads_tertiary = _shp_roads.apply(lambda row: row[_shp_roads['highway'].isin(['tertiary'])])

_shp_dst_roads_secondary = overlay(_shp_roads_secondary,_shp_district,how ="intersection")
_shp_dst_roads_primary = overlay(_shp_roads_primary,_shp_district,how ="intersection")
_shp_dst_roads_tertiary = overlay(_shp_roads_tertiary,_shp_district,how ="intersection")

# %%
_shp_roads_primary_secondary_tertiary = gpd.pd.concat([_shp_dst_roads_primary, _shp_dst_roads_secondary, _shp_dst_roads_tertiary])
_shp_roads_primary_secondary_tertiary.to_file(get_in_workdir('roads_prim_sec_ter.shp'))
_shp_roads_primary_secondary_tertiary = read_df_UT('workdir/roads_prim_sec_ter.shp')
_shp_roads_primary_secondary_tertiary.plot()

# %% [markdown]
# ### Railways

# %%
_shp_railways = read_df_UT("Railways_MP.shp")
_shp_railways.plot()

# %%
_shp_dst_railways = overlay(_shp_railways,_shp_district, how ="intersection")

# %%
_shp_dst_railways.plot()

# %%
_shp_dst_railways.to_file(get_in_workdir('railsways_dst.shp')) 

# %%
#_shp_dst_roads_secondary.plot()
#_shp_dst_roads_primary.plot()
_shp_dst_railways.plot()

# %% [markdown]
# ### Accessibility (Composite)
#
# This plot layers the primary and secondary roads with the railway lines in the district. Details of the library can be found here : https://geopandas.org/en/stable/docs/user_guide/mapping.html

# %%
fig4, ax4 = plt.subplots(figsize=(5, 5))

plot_common_features(fig4, ax4)
plot_cities(fig4, ax4)

_plt_district = _shp_district.plot(ax=ax4, figsize =(5,5),color="none",linewidth = 0.4)

_shp_dst_roads_primary.plot(color="#df2a29",label ="Primary roads",ax=ax4, linewidth=0.5)
_shp_dst_roads_secondary.plot(color="#E97C78",label ="Secondary roads",ax=ax4, linewidth=0.5)
_shp_dst_roads_tertiary.plot(color="#f0b8b3",label ="Tertiary roads",ax=ax4, linewidth=0.5)
_shp_dst_railways.plot(color="#da0404",label ="Railway roads",ax=ax4, linestyle='--', linewidth=0.5)

ax4.legend(loc='upper left', bbox_to_anchor=(0.8, 0.2), title = 'Legend\n', fontsize = 5.5, markerscale = 2, title_fontsize = 5.5, framealpha= 0, borderpad = 0.3, handletextpad = 0.5, handlelength = 1.0)

plt.savefig(get_in_workdir("Accessibility.jpg"),dpi =1500)
plt.show()

# %% [markdown]
# ## Evacuation

# %% [markdown]
# ### Powerlines

# %%
#_shp_powerlines = read_df_UT("Supporting_info/osm_powline_11.776643009779821_10.743913945502888_80.19273383153288_79.14689901832789.geojson")

# %%
#_shp_powerlines.info()
#_shp_powerlines.plot()

# %%
#_shp_dst_powerlines = overlay(_shp_powerlines, _shp_district, how='intersection')
#_shp_dst_powerlines.info()

# %% [markdown]
# ### Substations

# %%
_shp_substations = read_df_UT("list_substation_Panna.geojson")
_shp_substations.plot(color = 'black')

# %%
_shp_dst_substations_poly = overlay(_shp_substations, _shp_district, how='intersection')
_shp_dst_substations_poly.head()

# %%
_shp_dst_substations_epsg32644 = _shp_dst_substations_poly.to_crs(epsg = 32644)
_shp_dst_substations_epsg32644["geometry"] = _shp_dst_substations_epsg32644["geometry"].centroid
_shp_dst_substations = _shp_dst_substations_epsg32644.to_crs(4326)
_shp_dst_substations

# %% [markdown]
# ### Evacuation
#
# This plot layers the powerlines and substation locations in the district.
# Details of the library can be found here : https://geopandas.org/en/stable/docs/user_guide/mapping.html

# %%
fig5, ax5 = plt.subplots(figsize=(5, 5))

plot_common_features(fig5, ax5)
plot_cities(fig5, ax5)

# #Voltage annotation
# _shp_dst_powerlines['coords'] = _shp_dst_powerlines['geometry'].apply(lambda x: x.representative_point().coords[:])
# _shp_dst_powerlines['coords'] = [coords[0] for coords in _shp_dst_powerlines['coords']]
 
    
# _shp_dst_powerlines["coords"].tolist()
# _shp_dst_powerlines[['lat', 'lon']] = gpd.GeoDataFrame(_shp_dst_powerlines['coords'].tolist(), index=_shp_dst_powerlines.index)

# x = _shp_dst_powerlines["lat"]
# y = _shp_dst_powerlines["lon"]    
# labels =_shp_dst_powerlines["voltage"]

# for i in range(0,len(_shp_dst_powerlines), 4):
#     plt.text(x[i],y[i]+0.002,labels[i],fontsize=1.5,color = "red")


#Substations
_shp_dst_substations['coords'] = _shp_dst_substations['geometry'].apply(lambda x: x.representative_point().coords[:])
_shp_dst_substations['coords'] = [coords[0] for coords in _shp_dst_substations['coords']]

_shp_dst_substations["coords"].tolist()
_shp_dst_substations[['lat', 'lon']] = gpd.GeoDataFrame(_shp_dst_substations['coords'].tolist(), index=_shp_dst_substations.index)

x = _shp_dst_substations["lat"]
y = _shp_dst_substations["lon"]    
#labels =_shp_dst_substations["Capacity i"]

#for i in range(0,len(_shp_dst_substations)):
    #plt.text(x[i]+0.002,y[i],labels[i],fontsize=1.5,color = 'red', ha = 'left')
    

#plots
_plt_district = _shp_district.plot(ax=ax5, figsize =(5,5),color="none",linewidth = 0.5)
_shp_dst_substations.plot(color='red', marker='x', markersize=5, ax=ax5, linewidth=0.3, label='Substations')
#_shp_dst_powerlines.plot(color="#dd2c0e",ax =ax5, linewidth=0.3, linestyle='--', label='Transmission') 


ax5.legend(loc='upper left', bbox_to_anchor=(0.8, 0.2), title = 'Legend\n', fontsize = 5.5, markerscale = 2, title_fontsize = 5.5, framealpha= 0, borderpad = 0.3, handletextpad = 0.5, handlelength = 1.0)


print(plt.rcParams['font.family'])

plt.savefig(get_in_workdir("Substations.jpg"),dpi =1500) 
plt.show()

# %% [markdown]
# ## Settlements

# %%
_shp_settlemnts = read_df_UT('output_urban_cluster_1107.shp')
_shp_settlemnts.plot()

# %%
_shp_dst_settlemnts = overlay(_shp_settlemnts, _shp_district, how='intersection')
_shp_dst_settlemnts.plot()

# %%
fig6, ax6 = plt.subplots(figsize=(5, 5))

plot_common_features(fig6, ax6)
plot_cities(fig6, ax6)    

_plt_district = _shp_district.plot(ax=ax6, figsize =(5,5),color="none",linewidth = 0.5)
_shp_dst_settlemnts.plot(color="#e73429",ax =ax6, label = 'Built-up', zorder=3)

BU = mpatches.Patch(color="#e73429", label = 'Settlement')
plt.legend(handles = [BU], loc = 'upper left', bbox_to_anchor=(0.8, 0.2), title = 'Legend\n', fontsize = 5.5, markerscale = 2, title_fontsize = 5.5, framealpha= 0, borderpad = 0.3, handletextpad = 0.5, handlelength = 1.0)


plt.savefig(get_in_workdir("Settlements.jpg"),dpi =1500)
plt.show()

# %% [markdown]
# ## Landcover (shapefile)

# %%
shp_land_cover = read_df_UT('extrainputs/Panna_Landcover_shapefile.shp')
shp_dst_land_cover = overlay(shp_land_cover, _shp_district, how='intersection')
shp_dst_land_cover.to_file(get_in_workdir('land_cover_dst.shp'))

# %%
shp_dst_land_cover = read_df_UT('workdir/land_cover_dst.shp')
#shp_dst_land_cover.plot()

# %%
shp_dst_land_cover["DN"].unique()

# %%
shp_land_cover_builtup = shp_dst_land_cover[shp_dst_land_cover["DN"] == 6 ]
shp_land_cover_Barren = shp_dst_land_cover[shp_dst_land_cover["DN"] == 1 ]
shp_land_cover_sparseveg = shp_dst_land_cover[shp_dst_land_cover["DN"] == 2 ]
shp_land_cover_cropland  = shp_dst_land_cover[shp_dst_land_cover["DN"] == 3 ]
shp_land_cover_Forest = shp_dst_land_cover[shp_dst_land_cover["DN"] == 4 ]
shp_land_cover_Water = shp_dst_land_cover[shp_dst_land_cover["DN"] == 5 ]

# %%
fig7, ax7 = plt.subplots(figsize=(5, 5))

plot_common_features(fig7, ax7)
plot_cities(fig7, ax7)


_plt_district = _shp_district.plot(ax=ax7, figsize =(5,5),color="none",linewidth = 0.5)


shp_land_cover_Barren.plot(color="#424242",ax =ax7, label ='Unused')
shp_land_cover_sparseveg.plot(color="#f096ff",ax =ax7, label = 'Sparse vegetation')
shp_land_cover_cropland.plot(color="#18a558",ax =ax7, label = 'Cropland', zorder = 0)
shp_land_cover_Forest.plot(color="#B1D355",ax =ax7, label = 'Tree cover')
shp_land_cover_builtup.plot(color="#e73429",ax =ax7, label = 'Built-up')
shp_land_cover_Water.plot(color="#00B9F2",ax =ax7, label = 'Water')
#shp_land_cover_Water1.plot(color="#00B9F2",ax =ax7)

Unused = mpatches.Patch(color="#424242", label ='Unused')
SpV = mpatches.Patch(color="#f096ff", label = 'Sparse vegetation')
Crop = mpatches.Patch(color="#18a558", label = 'Cropland')
TC = mpatches.Patch(color="#B1D355", label = 'Tree cover')
BU = mpatches.Patch(color="#e73429", label = 'Built-up')
Water = mpatches.Patch(color="#00B9F2", label = 'Water')

    
plt.legend(handles = [Unused, SpV, Crop, TC, BU, Water], loc = 'upper left', bbox_to_anchor=(0.8, 0.2), title = 'Legend\n', fontsize = 5.5, markerscale = 2, title_fontsize = 5.5, framealpha= 0, borderpad = 0.3, handletextpad = 0.5, handlelength = 1.0)


plt.savefig(get_in_workdir("Landcover.jpg"),dpi =1500)
plt.show()

# %% [markdown]
# # Land suitability for solar

# %% [markdown]
# ### Technical Suitability - Technical, theoretical, and no potential lands

# %%
lc_tech_withE = read_df_UT('_ter_rdad_ar_tech/LC_Forest_final_area_mask_1_Panna.shp')
lc_tech_AtoE = lc_tech_withE.copy()

# %%
lc_tech_AtoE.shape

# %%
lc_tech = lc_tech_AtoE.drop(lc_tech_AtoE[lc_tech_AtoE["area_class"] == "E"].index)
lc_tech.shape

# %%
lc_tech.area_class.unique()

# %%
lc_tech.to_file(get_in_workdir('panna_forest_tech.shp'))

# %%
lc_tech = read_df_UT('workdir/panna_forest_tech.shp')
lc_tech.area_class.unique()

# %%
lc_theo = read_df_UT('_ter_rdad_th/LC_Forest_final_area_mask_1_Panna.shp')
lc_theo.area_class.unique()

# %%
shp_land_cover_Barren.to_file(get_in_workdir('Allbarren_Panna.shp'))
lc_barren = read_gpd_UT('workdir/Allbarren_Panna.shp')

# %%
lc_tech.shape, lc_theo.shape, lc_barren.shape

# %%
fig8, ax8 = plt.subplots(figsize=(5, 5))

plot_common_features(fig8, ax8)
plot_cities(fig8, ax8)


_shp_district.plot(figsize=(5,5),color="none", ax=ax8, linewidth = 0.5, zorder=5)

lc_barren.plot(color="#424242",ax =ax8)
lc_theo.plot(color="#15915C",ax =ax8)
lc_tech.plot(color="#99CC66",ax =ax8)


No_P = mpatches.Patch(color='#424242', label='No potential')
Theo_P = mpatches.Patch(color='#15915C', label='Theoretical potential')
Tech_P = mpatches.Patch(color='#99CC66', label='Technical potential')
    
plt.legend(handles = [No_P, Theo_P, Tech_P], loc = 'upper left', bbox_to_anchor=(0.8, 0.2), title = 'Legend\n', fontsize = 5.5, markerscale = 2, title_fontsize = 5.5, framealpha= 0, borderpad = 0.3, handletextpad = 0.5, handlelength = 1.0)


print(plt.rcParams['font.family'])


plt.savefig(get_in_workdir("Technical_suitability.jpg"),dpi =1500)
plt.show()

# %% [markdown]
# #### Files for Hyperlink

# %%
No_Potential = overlay(lc_barren, lc_theo, how='difference')
No_Potential.to_file(get_in_hyperlink('No_Potential'))
No_Potential.plot()

# %%
Only_theo = overlay(lc_theo, lc_tech, how='difference')
Only_theo.to_file(get_in_hyperlink('Only_theoretical'))
Only_theo.plot(color = '#15915C')

# %% [markdown]
# ### Technical lands distributed by size

# %%
lc_tech = read_df_UT('workdir/panna_forest_tech.shp')
#lc_tech.head()

# %%
lc_tech["area_class"].unique()

# %%
lc_tech_1to10 = lc_tech[lc_tech["area_class"] == "A"]
lc_tech_10to100 = lc_tech[lc_tech["area_class"] == "B"]
lc_tech_100to500 = lc_tech[lc_tech["area_class"] == "C"]
lc_tech_500 = lc_tech[lc_tech["area_class"] == "D"]

# %%
fig9, ax9 = plt.subplots(figsize=(5, 5))

plot_common_features(fig9, ax9)
plot_cities(fig9, ax9)


_shp_district.plot(figsize=(5,5),color="none", ax=ax9, linewidth = 0.5, zorder=5)

lc_tech_1to10.plot(color="#455555",ax =ax9, label='>1 to 10 ha')
lc_tech_10to100.plot(color="#157E5C",ax =ax9, label='>10 to 100 ha')
lc_tech_100to500.plot(color="#54AD64",ax =ax9, label='>100 to 500 ha')
lc_tech_500.plot(color="#99CC66",ax =ax9, label='>500 ha')


A_1to10 = mpatches.Patch(color='#455555', label='>1 to 10 ha')
B_10to100 = mpatches.Patch(color='#157E5C', label='>10 to 100 ha')
C_100to500 = mpatches.Patch(color='#54AD64', label='>100 acres')
D_500 = mpatches.Patch(color='#99CC66', label='>500 ha')
    
plt.legend(handles = [A_1to10, B_10to100, C_100to500, D_500], loc = 'upper left', bbox_to_anchor=(0.8, 0.2), title = 'Legend\n', fontsize = 5.5, markerscale = 2, title_fontsize = 5.5, framealpha= 0, borderpad = 0.3, handletextpad = 0.5, handlelength = 1.0)

print(plt.rcParams['font.family'])


plt.savefig(get_in_workdir("Distribution by size.jpg"),dpi =1500)
plt.show()

# %% [markdown]
# #### For hyperlink

# %%
lc_tech_1to10.to_file(get_in_hyperlink('Tech_1_to_10'))
lc_tech_10to100.to_file(get_in_hyperlink('Tech_10_to_100'))
lc_tech_100to500.to_file(get_in_hyperlink('Tech_100_to_500'))
lc_tech_500.to_file(get_in_hyperlink('Tech_min_500'))

#lc_tech_1to10.plot(), lc_tech_10to100.plot(), lc_tech_100to500.plot(), lc_tech_500.plot()

# %% [markdown]
# ### High Potential - High, medium, low potential lands

# %%
lc_high = read_df_UT('HastenVentures/High_Potential/High_Potential.shp')
lc_high.shape

# %%
lc_med = read_df_UT("HastenVentures/Medium_Potential/Medium_Potential.shp")
lc_med.shape

# %%
lc_low = read_df_UT('HastenVentures/Low_Potential/Low_Potential.shp')
lc_low.shape

# %%
#lc_tech_med = lc_tech.overlay(lc_med, how = 'difference')

# %%
#lc_tech_low = lc_tech_med.overlay(lc_high, how = 'difference')

# %%
#lc_tech_med.shape, lc_tech_low.shape

# %%
fig10, ax10 = plt.subplots(figsize=(5, 5))

plot_common_features(fig10, ax10)
plot_cities(fig10, ax10)

_shp_district.plot(figsize=(5,5),color="none", ax=ax10, linewidth = 0.5, zorder=5)


lc_low.plot(color="#21583B",ax =ax10, zorder = 3)
lc_med.plot(color="#009541",ax =ax10, zorder = 4)
lc_high.plot(color="#99CC66",ax =ax10, zorder = 5)


Low_P = mpatches.Patch(color='#21583B', label='Low potential')
Med_P = mpatches.Patch(color='#009541', label='Medium potential')
High_P = mpatches.Patch(color='#99CC66', label='High potential')
    
plt.legend(handles = [Low_P, Med_P, High_P], loc = 'upper left', bbox_to_anchor=(0.8, 0.2), title = 'Legend\n', fontsize = 5.5, markerscale = 2, title_fontsize = 5.5, framealpha= 0, borderpad = 0.3, handletextpad = 0.5, handlelength = 1.0)

print(plt.rcParams['font.family'])


plt.savefig(get_in_workdir("High Potential_H_M_L.jpg"),dpi =1500)
plt.show()

# %%

# %% [markdown]
# #### For hyperlink
#
# In this case, low, med, and high are not subgroups of one another, therefore the differently ranked lands do not repeat.

# %% [markdown]
# ## Competing Land Use

# %%
lc_tech = read_df_UT('workdir/panna_forest_tech.shp')

# %%
#lc_tech.head(), lc_tech['area_class'].unique()

# %%
_shp_water_high =  read_df_water_UT('_run_wd_lith_high/LC_Water_final.shp')
_shp_water_med = read_df_water_UT('_run_wd_lith_med/LC_Water_final.shp')

_shp_water_high.shape, _shp_water_med.shape

# %% [markdown]
# Forest Tech + Water High and Medium

# %%
water_high_dst = overlay(_shp_water_high,_shp_district,how ="intersection")
water_med_dst = overlay(_shp_water_med,_shp_district,how ="intersection")

water_high_dst.shape, water_med_dst.shape

# %%
#water_high_dst.head()

# %%
water_high_med_dst = gpd.pd.concat([water_high_dst, water_med_dst])
water_high_med_dst.shape

# %%
_shp_comp_use_forest_water = overlay(water_high_med_dst,lc_tech, how='intersection')
#_shp_comp_use_forest_water.plot()
#water_high_med_dst.plot()
#lc_tech.plot()

# %% [markdown]
# Competing land use plot

# %%
fig11, ax11 = plt.subplots(figsize=(5, 5))

plot_common_features(fig11, ax11)
plot_cities(fig11, ax11)

_shp_district.plot(figsize=(5,5),color="none", ax=ax11, linewidth = 0.5, zorder=5)

lc_tech.plot(color="#B1D355",ax =ax11)
water_high_med_dst.plot(color="#00B9F2",ax =ax11)
_shp_comp_use_forest_water.plot(color="#e73429",ax =ax11, zorder = 6)


F = mpatches.Patch(color='#B1D355', label='Forest')
W = mpatches.Patch(color='#00B9F2', label='Water')
FnW = mpatches.Patch(color='#e73429', label='Forest and water')
    
plt.legend(handles = [F, W, FnW], loc = 'upper left', bbox_to_anchor=(0.8, 0.2), title = 'Legend\n', fontsize = 5.5, markerscale = 2, title_fontsize = 5.5, framealpha= 0, borderpad = 0.3, handletextpad = 0.5, handlelength = 1.0)

print(plt.rcParams['font.family'])

plt.savefig(get_in_workdir("Competing Land Use.jpg"),dpi =1500)
plt.show()

# %% [markdown]
# #### For hyperlink

# %%
water_high_med_dst.to_file(get_in_hyperlink('Water_med_high_dst'))
_shp_comp_use_forest_water.to_file(get_in_hyperlink('Cometing_lands_f_w'))

# %%

# %%

# %%
_shp_district = read_df_UT32644('extrainputs/panna_dst.shp')

# %%
_shp_district['area_hect']=(_shp_district.geometry.area)/10**4

# %%
_shp_district = _shp_district.to_crs(4326)

# %%
_shp_district.to_file(get_in_HV('District_Boundary'))

# %%

# %%
shp_dst_land_cover = read_df_UT32644('workdir/land_cover_dst.shp')

# %%
shp_dst_land_cover

# %%
shp_dst_land_cover['area_hect']=(shp_dst_land_cover.geometry.area)/10**4

# %%
shp_dst_land_cover = shp_dst_land_cover.to_crs(4326)

# %%
shp_dst_land_cover

# %%
shp_dst_land_cover['DN'].unique()


# %%
def lc_layer(shp_dst_land_cover):
    if shp_dst_land_cover['DN'] == 1:
        return 'unused'
    elif shp_dst_land_cover['DN'] == 2:
        return 'sparse vegetation'
    elif shp_dst_land_cover['DN'] == 3:
        return 'cropland'
    elif shp_dst_land_cover['DN'] == 4:
        return 'tree cover'
    elif shp_dst_land_cover['DN'] == 5:
        return 'water'
    elif shp_dst_land_cover['DN'] == 6:
        return 'builtup'
    else:
        return 0
    
shp_dst_land_cover['Type'] = shp_dst_land_cover.apply(lc_layer, axis =1)

# %%

# %%
shp_dst_land_cover.to_file(get_in_HV('Landcover'))

# %%
shp_land_cover_builtup = shp_dst_land_cover[shp_dst_land_cover["DN"] == 6 ]
shp_land_cover_Barren = shp_dst_land_cover[shp_dst_land_cover["DN"] == 1 ]
shp_land_cover_sparseveg = shp_dst_land_cover[shp_dst_land_cover["DN"] == 2 ]
shp_land_cover_cropland  = shp_dst_land_cover[shp_dst_land_cover["DN"] == 3 ]
shp_land_cover_Forest = shp_dst_land_cover[shp_dst_land_cover["DN"] == 4 ]
shp_land_cover_Water = shp_dst_land_cover[shp_dst_land_cover["DN"] == 5 ]

# %%
shp_land_cover_builtup.to_file(get_in_HV('Builtup'))
shp_land_cover_Barren.to_file(get_in_HV('Unused'))
shp_land_cover_sparseveg.to_file(get_in_HV('SparseVegetation'))
shp_land_cover_cropland.to_file(get_in_HV('Cropland'))
shp_land_cover_Forest.to_file(get_in_HV('Treecover'))
shp_land_cover_Water.to_file(get_in_HV('Water'))

# %%
shp_land_cover_Water.shape

# %%
_shp_dst_water =_shp_dst_water.to_crs(32644)

# %%
_shp_dst_water['area_hect']=(_shp_dst_water.geometry.area)/10**4

# %%
_shp_dst_water = _shp_dst_water.to_crs(4326)

# %%
_shp_dst_water.to_file(get_in_HV('Water_district'))

# %%

# %%

# %%
_shp_dst_roads_secondary = overlay(_shp_roads_secondary,_shp_district,how ="intersection")
_shp_dst_roads_primary = overlay(_shp_roads_primary,_shp_district,how ="intersection")
_shp_dst_roads_tertiary = overlay(_shp_roads_tertiary,_shp_district,how ="intersection")

# %%
_shp_dst_roads_secondary.plot()

# %%
_shp_dst_roads_primary.to_file(get_in_HV('Primary_roads'))
_shp_dst_roads_secondary.to_file(get_in_HV('Secondary_roads'))
_shp_dst_roads_tertiary.to_file(get_in_HV('Tertiary_roads'))

# %%
_shp_dst_railways = overlay(_shp_railways,_shp_district, how ="intersection")

# %%
_shp_dst_railways.to_file(get_in_HV('Railway_lines')) 

# %%

# %%
_shp_dst_substations.head() 

# %%
_shp_dst_substations = _shp_dst_substations.drop(columns=['coords'])

# %%
_shp_dst_substations.to_file(get_in_HV('Substations')) 

# %%

# %%
_shp_dst_forest_reserve = overlay(_shp_forest_reserve, _shp_district, how='intersection')

# %%
_shp_dst_forest_reserve =_shp_dst_forest_reserve.to_crs(32644)

# %%
_shp_dst_forest_reserve['area_hect']=(_shp_dst_forest_reserve.geometry.area)/10**4

# %%
_shp_dst_forest_reserve = _shp_dst_forest_reserve.to_crs(4326)

# %%
_shp_dst_forest_reserve.to_file(get_in_HV('Forest_reserve')) 

# %%

# %%
_shp_dst_settlemnts = overlay(_shp_settlemnts, _shp_district, how='intersection')

# %%
_shp_dst_settlemnts.crs

# %%
_shp_dst_settlemnts =_shp_dst_settlemnts.to_crs(32644)

# %%
_shp_dst_settlemnts['area_hect']=(_shp_dst_settlemnts.geometry.area)/10**4

# %%
_shp_dst_settlemnts = _shp_dst_settlemnts.to_crs(4326)

# %%
_shp_dst_settlemnts.to_file(get_in_HV('Settlements')) 

# %%

# %%
lc_tech = read_df_UT('workdir/panna_forest_tech.shp')

# %%
lc_tech.to_file(get_in_HV('Technical_Potential'))

# %%

# %%
lc_theo = read_df_UT('_ter_rdad_th/LC_Forest_final_area_mask_1_Panna.shp')

# %%
lc_theo.to_file(get_in_HV('Theoretical_Potential'))

# %%

# %%
No_Potential = overlay(lc_barren, lc_theo, how='difference')

# %%
No_Potential = No_Potential.to_crs(32644)

# %%
No_Potential['area_hect']=(No_Potential.geometry.area)/10**4

# %%
No_Potential = No_Potential.to_crs(4326)

# %%
No_Potential.to_file(get_in_HV('No_Potential')) 

# %%

# %%
lc_med = read_df_UT("_ter_elev_watpot_wtad_rdad_urad_ar_med1/LC_Forest_final_area_mask_1_Panna.shp")

# %%
lc_med.shape

# %%
lc_med = lc_med.to_crs(32644)

# %%
lc_med['area_hect']=(lc_med.geometry.area)/10**4

# %%
lc_med = lc_med.to_crs(4326)


# %%
def areaclass(lc_med):
    if (lc_med['area_hect']>1) & (lc_med['area_hect'] <= 10):
        return 'A'
    elif (lc_med['area_hect'] > 10) & (lc_med['area_hect'] <= 100):
        return 'B'
    elif (lc_med['area_hect'] > 100) & (lc_med['area_hect'] <= 500):
        return 'C'
    elif lc_med['area_hect'] > 500:
        return 'D'
    else:
        return 'E'
    
lc_med['area_class'] = lc_med.apply(areaclass, axis =1)

# %%
lc_med = lc_med.drop(lc_med[lc_med["area_class"] == "E"].index)

# %%
lc_med.to_file(get_in_HV('Medium_Potential'))

# %%
lc_med.head()

# %%
lc_tech.head()

# %%
lc_tech_med = lc_tech.overlay(lc_med, how = 'difference')

# %%
lc_tech.shape, lc_med.shape, lc_tech_med.shape

# %%
lc_tech.geometry.area.sum()

# %%
lc_med.geometry.area.sum()

# %%
lc_tech_med.geometry.area.sum()

# %%
lc_tech.area_hect.sum()

# %%
lc_tech_med.area_hect.sum()

# %%
lc_high = read_df_UT('_ter_fcor_elev_watpot_wtad_rdad_urad_ar_high1/LC_Forest_final_area_mask_1_Panna.shp')

# %%
lc_high.shape

# %%
lc_high = lc_high.to_crs(32644)

# %%
lc_high['area_hect']=(lc_high.geometry.area)/10**4

# %%
lc_high = lc_high.to_crs(4326)


# %%
def areaclass(lc_high):
    if (lc_high['area_hect']>1) & (lc_high['area_hect'] <= 10):
        return 'A'
    elif (lc_high['area_hect'] > 10) & (lc_high['area_hect'] <= 100):
        return 'B'
    elif (lc_high['area_hect'] > 100) & (lc_high['area_hect'] <= 500):
        return 'C'
    elif lc_high['area_hect'] > 500:
        return 'D'
    else:
        return 'E'
    
lc_high['area_class'] = lc_high.apply(areaclass, axis =1)

# %%
lc_high.area_class.unique()

# %%
lc_high = lc_high.drop(lc_high[lc_high["area_class"] == "E"].index)

# %%
lc_high.to_file(get_in_HV('High_Potential'))

# %%

# %%
lc_tech_low = lc_tech_med.overlay(lc_high, how = 'difference')

# %%
lc_tech_low.geometry.area.sum(), lc_high.geometry.area.sum(), lc_tech_med.geometry.area.sum()

# %%
lc_tech_low.shape

# %%
lc_tech_low = lc_tech_low.to_crs(32644)

# %%
lc_tech_low['area_hect']=(lc_tech_low.geometry.area)/10**4

# %%
lc_tech_low = lc_tech_low.to_crs(4326)

# %%
lc_tech_low.area_hect.max()


# %%
def areaclass(lc_tech_low):
    if (lc_tech_low['area_hect']>1) & (lc_tech_low['area_hect'] <= 10):
        return 'A'
    elif (lc_tech_low['area_hect'] > 10) & (lc_tech_low['area_hect'] <= 100):
        return 'B'
    elif (lc_tech_low['area_hect'] > 100) & (lc_tech_low['area_hect'] <= 500):
        return 'C'
    elif lc_tech_low['area_hect'] > 500:
        return 'D'
    else:
        return 'E'
    
lc_tech_low['area_class'] = lc_tech_low.apply(areaclass, axis =1)

# %%
lc_tech_low.area_class.unique()

# %%
lc_tech_low.to_file(get_in_HV('Low_Potential'))

# %%
lc_tech_low.area_hect.sum()+lc_med.area_hect.sum()+lc_high.area_hect.sum(), lc_tech.area_hect.sum()

# %%

# %%
water_high_med_dst.shape

# %%
water_high_med_dst = water_high_med_dst.to_crs(32644)

# %%
water_high_med_dst['area_hect']=(water_high_med_dst.geometry.area)/10**4

# %%
water_high_med_dst = water_high_med_dst.to_crs(4326)

# %%
#water_high_med_dst.head()

# %%
water_high_med_dst.to_file(get_in_HV('WaterHarvesting_Potential'))

# %%

# %%
_shp_comp_use_forest_water.crs

# %%
_shp_comp_use_forest_water = _shp_comp_use_forest_water.to_crs(32644)

# %%
_shp_comp_use_forest_water['area_hect']=(_shp_comp_use_forest_water.geometry.area)/10**4

# %%
_shp_comp_use_forest_water = _shp_comp_use_forest_water.to_crs(4326)

# %%
_shp_comp_use_forest_water.drop(columns=['area_class'])

# %%
_shp_comp_use_forest_water.to_file(get_in_HV('CompetingUse_w_Water'))

# %%

# %%
high1 = read_df_UT('_ter_fcor_elev_watpot_wtad_rdad_urad_ar_high1/LC_Forest_final_area_mask_1_Panna.shp')
med1 = read_df_UT('_ter_elev_watpot_wtad_rdad_urad_ar_med1/LC_Forest_final_area_mask_1_Panna.shp')
#settlement = read_df_UT('output_urban_cluster_1107.shp')

# %%
fig13, ax13 = plt.subplots(figsize=(5, 5))

plot_common_features(fig13, ax13)
plot_cities(fig13, ax13)

_shp_district.plot(figsize=(5,5),color="none", ax=ax13, linewidth = 0.5, zorder=5)


_shp_dst_settlemnts.plot(color="#e73429",ax =ax13, zorder = 3)
med1.plot(color="#009541",ax =ax13, zorder = 4)
high1.plot(color="#99CC66",ax =ax13, zorder = 5)


Low_P = mpatches.Patch(color='#e73429', label='Settlements')
Med_P = mpatches.Patch(color='#009541', label='Medium potential')
High_P = mpatches.Patch(color='#99CC66', label='High potential')
    
plt.legend(handles = [Low_P, Med_P, High_P], loc = 'upper left', bbox_to_anchor=(0.8, 0.2), title = 'Legend\n', fontsize = 5.5, markerscale = 2, title_fontsize = 5.5, framealpha= 0, borderpad = 0.3, handletextpad = 0.5, handlelength = 1.0)

print(plt.rcParams['font.family'])


plt.savefig(get_in_workdir("Settlement_prox.jpg"),dpi =1500)
plt.show()

# %%

# %%

# %%
