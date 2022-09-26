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

# %% [markdown]
# ## Imports

# %%
import sys, subprocess,time
import pandas as pd
import numpy as np
import geopandas as gpd
from pyproj import CRS
from shapely.ops import unary_union, polygonize
from rtree import index
from shapely.geometry import Polygon
from sklearn.cluster import DBSCAN
import rasterio
#import rasterstats as rs
import glob
from shapely.affinity import rotate
from shapely.geometry import LineString, Point, MultiPolygon,MultiLineString 
from shapely.ops import unary_union, polygonize
from geopandas import overlay
import shapely.wkt
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib_scalebar.scalebar import ScaleBar

from pathlib import Path


# %% [markdown]
# ## Input and Working Dir paths
#
# Since inputs data may be read-only, we separate path abstractions for inputs and working directories. The path are wrapped in function calls, so this is the only cell which needs to be modified when the notebook runtime folders are changed.

# %%
def get_input(stem):
    return "D:/LiLa_Nagapattinam/" + stem

def get_in_workdir(stem):
    if _is_kaggle or _is_colab:
        return './' + stem
    else:
        return get_input('workdir/' + stem)

def read_df_UT(stem):
    return gpd.read_file(get_input(stem)).to_crs(epsg = 4326)



# %% [markdown]
# # Creating Visuals for Lila Solar Report

# %% [markdown]
# ## District shape

# %% tags=[]
_shp_district = read_df_UT("Practice/Nagapattinam_proj32644.shp")
_shp_district.info()

# %% [markdown]
# ## Powerlines

# %%
_shp_powerlines = read_df_UT("Supporting_info/osm_powline_11.776643009779821_10.743913945502888_80.19273383153288_79.14689901832789.geojson")

# %% tags=[]
_shp_powerlines.info()

# %% tags=[]
_shp_powerlines.plot(color = _shp_powerlines["color"])
plt.show()

# %% tags=[]
_shp_dst_powerlines = overlay(_shp_powerlines, _shp_district, how='intersection')
_shp_dst_powerlines.info()

# %% [markdown]
# ## Substations

# %%
_shp_substations = read_df_UT("Supporting_info/list_substation_TN_corr.geojson/list_substation_TN_corr.shp")
_shp_dst_substations = overlay(_shp_substations, _shp_district, how='intersection')

# %% [markdown]
# ## Composite plot - Substation + Powerlines
#
# This plot layers the powerlines and substation locations in the district.
# Details of the library can be found here : https://geopandas.org/en/stable/docs/user_guide/mapping.html

# %%
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

# %% tags=[]


_plt_district = _shp_district.plot(figsize=(5,5),color="none")
_shp_dst_powerlines.plot(color="grey",ax =_plt_district) 
_shp_dst_substations.plot(color="black",ax=_plt_district)  

_plt_district.xaxis.tick_top()

plt.xlim(79.40,80.00) 
plt.ylim(10.93,11.45)  

plt.grid(color="grey",linestyle = '--', linewidth = 0.5) 

_plt_district.tick_params(axis='x', colors='grey',labelsize=5) 
_plt_district.tick_params(axis='y', colors='grey',labelsize=5)

_plt_district.xaxis.set_major_formatter(FormatStrFormatter('%.2f')) 
_plt_district.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

_plt_district.spines['bottom'].set_color('lightgrey') 
_plt_district.spines['top'].set_color('lightgrey') 
_plt_district.spines['right'].set_color('lightgrey')
_plt_district.spines['left'].set_color('lightgrey')

plt.text(79.58805,11.08840,s ="Kadalangudy",fontsize="x-small")
plt.text(79.61593,11.20177,s ="Manalmedu",fontsize="x-small")
plt.text(79.66633,11.10438,s ="Mayiladuthurai",fontsize="x-small")
plt.text(79.80977,11.17567,s ="Thiruvengadu",fontsize="x-small")
plt.text(79.72628,11.19915,s ="Vaitheeswarankoil",fontsize="x-small")



plt.savefig(get_in_workdir("powerlines.jpg"),dpi =1500) 
plt.show()

# %% tags=[]
_shp_dst_substations.head()

# %% [markdown]
# ## Roads - Primary and Secondary 

# %%
_shp_roads = read_df_UT("Supporting_info/output_osmroad_edges_11.776643009779821_10.743913945502888_80.19273383153288_79.14689901832789.geojson/edges.shp")
_shp_roads.plot()

# %% tags=[]
_shp_roads.info()

# %% tags=[]
_shp_roads.head()

# %%
_shp_roads.shape

# %% tags=[]
_shp_roads["highway"].unique()

# %%
_shp_roads_secondary = _shp_roads.apply(lambda row: row[_shp_roads['highway'].isin(['secondary'])])
_shp_roads_primary = _shp_roads.apply(lambda row: row[_shp_roads['highway'].isin(['primary'])])

_shp_dst_roads_secondary = overlay(_shp_roads_secondary,_shp_district,how ="intersection")
_shp_dst_roads_primary = overlay(_shp_roads_primary,_shp_district,how ="intersection")

# %% [markdown]
# ## Railways

# %%
_shp_railways = read_df_UT("Supporting_info/railway/Railways.shp")
_shp_railways.plot()

# %%
_shp_dst_railways = overlay(_shp_railways,_shp_district,how ="intersection")

# %% tags=[]
_shp_dst_roads_secondary.plot()
_shp_dst_roads_primary.plot()
_shp_dst_railways.plot()

# %% [markdown]
# ## Composite plot - Roads (Primary & Secondary) + Railways
#
# This plot layers the primary and secondary roads with the railway lines in the district.
# Details of the library can be found here : https://geopandas.org/en/stable/docs/user_guide/mapping.html

# %%
_plt_district = _shp_district.plot(figsize=(5,5),color="none",zorder=3)
_shp_dst_roads_secondary.plot(color="grey",label ="Secondary roads",ax =_plt_district)
_shp_dst_roads_primary.plot(color="brown",label ="Primary roads",ax=_plt_district)
_shp_dst_railways.plot(color="black",label ="Railway roads",ax=_plt_district)




plt.xlim(79.40,80.00)
plt.ylim(10.93,11.45)

plt.grid(color="grey",linestyle = '--', linewidth = 0.5)

_plt_district.tick_params(axis='x', colors='grey', labelsize=3,labeltop=True,labelrotation =270)
_plt_district.tick_params(axis='y', colors='grey', labelsize=3,labelright=True,labelrotation =0)

plt.text(79.58805,11.08840,s ="Kadalangudy",fontsize=4,color = "grey")
plt.text(79.61593,11.20177,s ="Manalmedu",fontsize=4,color = "grey")
plt.text(79.66633,11.10438,s ="Mayiladuthurai",fontsize=4,color = "grey")
plt.text(79.80977,11.17567,s ="Thiruvengadu",fontsize=4,color = "grey")
plt.text(79.72628,11.19915,s ="Vaitheeswarankoil",fontsize=4,color = "grey")

fontprops = fm.FontProperties(size=5.5)


scalebar = AnchoredSizeBar(_plt_district.transData,
                           0.1, '11 km', 'lower left', 
                           pad=0.005,
                           color='lightgrey',
                           frameon=False,
                           size_vertical=0.005,
                           fontproperties=fontprops)

_plt_district.add_artist(scalebar)

x, y, arrow_length = 0.5, 0.99, 0.1
_plt_district.annotate('N',color= "lightgrey", xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='none',edgecolor="lightgrey", width=4, headwidth=12),
            ha='center', va='center', fontsize=10,
            xycoords=_plt_district.transAxes)

_plt_district.spines['bottom'].set_color('none')
_plt_district.spines['top'].set_color('none') 
_plt_district.spines['right'].set_color('none')
_plt_district.spines['left'].set_color('none')

plt.savefig(get_in_workdir("roadway.jpg"),dpi =1500)
plt.show()

# %% [markdown]
# ## Water Bodies

# %%
_shp_water = read_df_UT("Practice/water_TN.shp")
_shp_dst_water = overlay(_shp_water,_shp_district,how ="intersection")
_shp_water.plot()

# %% [markdown]
# This plot shows the water bodies within the boundaries of the district.
# Details of the library can be found here : https://geopandas.org/en/stable/docs/user_guide/mapping.html

# %%
_shp_dst_water["color"] = "Waterbodies"

# %%
_shp_dst_water["GEOMORPHOL"].unique()

# %%
plt.rcParams['font.family'] = 'Helvetica'

_plt_district = _shp_district.plot(figsize =(5,5),color="none",linewidth = 0.5)

_shp_dst_water.plot(column='color', categorical=True, legend=True,legend_kwds={'loc': 'lower right','title':'Legend',"fontsize": 5.5,'markerscale':0.5,'title_fontsize':5.5,"framealpha":0,"borderpad":0.3,"handletextpad":0.5,"handlelength":1.0},ax =_plt_district)
_shp_dst_substations.plot(color="black",ax=_plt_district,markersize= 3)  


plt.xlim(79.40,80.00)
plt.ylim(10.93,11.45)

plt.grid(color="grey",linestyle = '--', linewidth = 0.10)

_plt_district.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
_plt_district.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

_plt_district.tick_params(axis='x', colors='grey', labelsize=3,labeltop=True,labelrotation =270)
_plt_district.tick_params(axis='y', colors='grey', labelsize=3,labelright=True,labelrotation =0)

plt.text(79.58805,11.08840,s ="Kadalangudy",fontsize=4,color = "grey")
plt.text(79.61593,11.20177,s ="Manalmedu",fontsize=4,color = "grey")
plt.text(79.66633,11.10438,s ="Mayiladuthurai",fontsize=4,color = "grey")
plt.text(79.80977,11.17567,s ="Thiruvengadu",fontsize=4,color = "grey")
plt.text(79.72628,11.19915,s ="Vaitheeswarankoil",fontsize=4,color = "grey")

fontprops = fm.FontProperties(size=5.5)


scalebar = AnchoredSizeBar(_plt_district.transData,
                           0.1, '10 km', 'lower left', 
                           pad=0.005,
                           color='lightgrey',
                           frameon=False,
                           size_vertical=0.005,
                           fontproperties=fontprops)

_plt_district.add_artist(scalebar)

x, y, arrow_length = 0.5, 0.99, 0.1
_plt_district.annotate('N',color= "lightgrey", xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='none',edgecolor="lightgrey", width=4, headwidth=12),
            ha='center', va='center', fontsize=10,
            xycoords=_plt_district.transAxes)

_plt_district.spines['bottom'].set_color('none')
_plt_district.spines['top'].set_color('none') 
_plt_district.spines['right'].set_color('none')
_plt_district.spines['left'].set_color('none')
# plt.legend()
# plt.rcParams['font.family'] = 'Helvetica'
# plt.rcParams.update({'font.family':'sans-serif'})
# plt.rcParams.update({'font.sans-serif':'Helvetica'})
# plt.rcdefaults()
plt.savefig(get_in_workdir("water.jpg"),dpi =1500)
plt.show()
print(plt.rcParams['font.family'])


# %%
def read_raster_UT_path(path):
    return rasterio.open(path)

def read_raster_UT(stem):
    return read_raster_UT_path(get_input(stem))


# %%
raster = read_raster_UT("Supporting_info/GHI_Nagapattinam.tif")

# %%
raster.meta

# %% tags=[]
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

# %%
raster.crs

# %%
dstCrs = {'init': 'EPSG:4326'}

# %%
#calculate transform array and shape of reprojected raster
transform, width, height = calculate_default_transform(
        raster.crs, dstCrs, raster.width, raster.height, *raster.bounds)

raster.transform
transform

# %%
#working of the meta for the destination raster
kwargs = raster.meta.copy()
kwargs.update({
        'crs': dstCrs,
        'transform': transform,
        'width': width,
        'height': height,
    })

# %%
# def open_raster_to_write(fname, kwargs):
#     dstPath = get_in_workdir(fname)
#     if os.path.exists(dstPath):
#         os.remove(dstPath)
#     return rasterio.open(dstPath, 'w', **kwargs)

# dstRst = open_raster_to_write('GHIepsg_Nagapattinam.tif', kwargs)

# %%
# dstRst.meta

# %%
# #reproject and save raster band data
# for i in range(1, raster.count + 1):
#     reproject(
#         source=rasterio.band(raster, i),
#         destination=rasterio.band(dstRst, i),
#         #src_transform=srcRst.transform,
#         src_crs=raster.crs,
#         #dst_transform=transform,
#         dst_crs=dstCrs,
#         resampling=Resampling.nearest)

# #close destination raster
# dstRst.close()        

# %%
import rasterio
from rasterio.plot import show
dstRst = read_raster_UT_path(get_in_workdir('GHIepsg_Nagapattinam.tif'))
show(dstRst)

# %% [markdown]
# land cover high, med, low

# %%
lc_high = read_df_UT('solar/_rl_elev_rd_wat_co_trans_ar_sub_rdpx_trsub_trat_subat_rdat_ir_high/LC_Solar_final_area_mask_1_Nagapattinam.shp')

# %%
lc_med = read_df_UT("solar/_rl_elev_rd_wat_co_trans_ar_sub_rdpx_trsub_trat_subat_rdat_ir_medatt/LC_Solar_final_area_mask_1_Nagapattinam.shp")

# %%
lc_low = read_df_UT('solar/_rl_elev_rd_wat_trans_ar_sub_rdpx_trsub_low/LC_Solar_final_area_mask_1_Nagapattinam.shp')

# %%
merge_lc1 = overlay(lc_high,_shp_district,how ="intersection")
merge_lc2 = overlay(lc_med,_shp_district,how ="intersection")
merge_lc3 = overlay(lc_low,_shp_district,how ="intersection")


# %%
merge_lc3.head()

# %%
ax = _shp_district.plot(figsize=(5,5),color="none",zorder=3)
merge_lc3.plot(color="blue",ax =ax)
merge_lc2.plot(color="green",ax =ax)
merge_lc1.plot(color="red",ax =ax)

# ax.xaxis.tick_top()
plt.xlim(79.40,80.00)
plt.ylim(10.93,11.45)

# plt.grid(color="grey",linestyle = '--', linewidth = 0.50)

# plt.ticklabel_format(axis="both")
# ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# ax.tick_params(axis='x', colors='grey', labelsize=5)
# ax.tick_params(axis='y', colors='grey', labelsize=5)

# ax.spines['bottom'].set_color('lightgrey')
# ax.spines['top'].set_color('lightgrey') 
# ax.spines['right'].set_color('lightgrey')
# ax.spines['left'].set_color('lightgrey')

plt.axis('off')
plt.savefig(get_in_workdir("land_cover.jpg"),dpi =1500)
plt.show()

# %%
lc_tech = read_df_UT('solar/_rl_elev_rd_wat_co_trans_ar_sub_rdpx_trsub_tech/LC_Solar_final_area_mask_1_Nagapattinam.shp')

# %%
lc_theo = read_df_UT('solar/_rl_elev_rd_wat_co_th/LC_Solar_final_mask_val_1_Nagapattinam.shp')

# %%
lc_barren = read_df_UT('solar/all_lands_barren/all_BarrenLands_Mayu.shp')

# %%
lc_tech.plot()
lc_theo.plot()
lc_barren.plot()

# %%
_plt_district = _shp_district.plot(figsize=(5,5),color="none",zorder=3)
x = lc_barren.plot(color="brown",ax =_plt_district)
y = lc_theo.plot(color="green",ax =_plt_district)
z = lc_tech.plot(color="blue",ax =_plt_district)
_plt_district.xaxis.tick_top()
_plt_district.xaxis.tick_top()

plt.xlim(79.40,80.00)
plt.ylim(10.93,11.45)

plt.grid(color="grey",linestyle = '--', linewidth = 0.50)

_plt_district.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
_plt_district.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

_plt_district.tick_params(axis='x', colors='grey', labelsize=5)
_plt_district.tick_params(axis='y', colors='grey', labelsize=5)

plt.text(79.58805,11.08840,s ="Kadalangudy",fontsize="xx-small")
plt.text(79.61593,11.20177,s ="Manalmedu",fontsize="xx-small")
plt.text(79.66633,11.10438,s ="Mayiladuthurai",fontsize="xx-small")
plt.text(79.80977,11.17567,s ="Thiruvengadu",fontsize="xx-small")
plt.text(79.72628,11.19915,s ="Vaitheeswarankoil",fontsize="xx-small")

fontprops = fm.FontProperties(size=10)


scalebar = AnchoredSizeBar(_plt_district.transData,
                           0.1, '11 km', 'lower left', 
                           pad=0.005,
                           color='Black',
                           frameon=False,
                           size_vertical=0.005,
                           fontproperties=fontprops)

_plt_district.add_artist(scalebar)

x, y, arrow_length = 0.9, 0.99, 0.1
_plt_district.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='none', width=8, headwidth=17),
            ha='center', va='center', fontsize=10,
            xycoords=_plt_district.transAxes)

_plt_district.spines['bottom'].set_color('none')
_plt_district.spines['top'].set_color('none') 
_plt_district.spines['right'].set_color('none')
_plt_district.spines['left'].set_color('none')

# plt.axis('off')
plt.savefig(get_in_workdir("barren_tech_theo.jpg"),dpi =1500)
plt.show()

# %%
lc_tech.head()

# %%
lc_tech["area_class"].unique()

# %%
lc_tech_A = lc_tech[lc_tech["area_class"] == "A"]
lc_tech_B = lc_tech[lc_tech["area_class"] == "B"]
lc_tech_C = lc_tech[lc_tech["area_class"] == "C"]

# %%
ax =_shp_district.plot(_shp_district,figsize=(5,5),color="none",zorder=3)
x = lc_tech_A.plot(color="blue",ax =ax)
y = lc_tech_B.plot(color="green",ax =ax)
z = lc_tech_C.plot(color="red",ax =ax)
plt.axis('off')
plt.savefig(get_in_workdir("tech_a_b_c.jpg"),dpi =1500)
plt.show()

# %%
shp_land_cover = read_df_UT('solar/Trial.geojson')

# %%
shp_land_cover.head()

# %%
shp_land_cover["DN"].unique()

# %% [markdown]
# shape files 

# %%
shp_land_cover_buildup = shp_land_cover[shp_land_cover["DN"] == 6 ]
shp_land_cover_Barren = shp_land_cover[shp_land_cover["DN"] == 1 ]
shp_land_cover_sparseveg = shp_land_cover[shp_land_cover["DN"] == 2 ]
shp_land_cover_cropland  = shp_land_cover[shp_land_cover["DN"] == 3 ]
shp_land_cover_Forest = shp_land_cover[shp_land_cover["DN"] == 4 ]
shp_land_cover_Water = shp_land_cover[shp_land_cover["DN"] == 5 ]

# %%
ax =_shp_district.plot(_shp_district,figsize=(5,5),color="none")
shp_land_cover_Barren.plot(color="orange",ax =ax)
shp_land_cover_sparseveg.plot(color="black",ax =ax)
shp_land_cover_cropland.plot(color="brown",ax =ax,alpha = 0.5)
shp_land_cover_Forest.plot(color="green",ax =ax)
shp_land_cover_buildup.plot(color="red",ax =ax)
shp_land_cover_Water.plot(color="skyblue",ax =ax)
plt.axis('off')
plt.savefig(get_in_workdir("all 6 categories.jpg"),dpi =1500)
plt.show()

# %%
shp_land_cover_buildup.plot()

# %%
shp_land_cover_buildup.head()

# %%
shp_land_cover_buildup.shape

# %%
ax =_shp_district.plot(_shp_district,figsize=(5,5),color="none",zorder = 0)
shp_land_cover_buildup.plot(color="red",ax =ax)
plt.axis('off')
plt.savefig(get_in_workdir("buildup_area.jpg"),dpi =1500)
plt.show()

# %%
_shp_water_high =  read_df_UT('water/_wd_run_high/LC_Water_final.shp')

# %%
_shp_water_high.shape

# %%
tech_high_med = gpd.pd.concat([lc_tech_B,lc_tech_C])


# %%
tech_high_med.head()

# %%
tech_high_med["area_class"].unique()

# %%
tech_high_med.plot(cmap ="Accent")

# %%
tech_high_med_dist = overlay(tech_high_med,_shp_district,how ="intersection")

# %%
tech_high_med_dist.plot()

# %%
_shp_water_high =  read_df_UT('water/_wd_run_high/LC_Water_final.shp')

# %%
_shp_water_med = read_df_UT('water/_wd_run_med/LC_Water_final.shp')

# %%
forest_med = read_df_UT('forest/_ter_elev_watpot_ar_med/LC_Forest_final_area_mask_1_Nagapattinam.shp')

# %%
water_high_med = gpd.pd.concat([_shp_water_high,_shp_water_med])

# %%
water_high_med_dist = overlay(water_high_med,_shp_district,how ="intersection")

# %%
water_high_med_dist.shape

# %%
water_high_med_dist.head()

# %%
_shp_merge_tech_water = overlay(water_high_med_dist,tech_high_med_dist, how='intersection')

# %%
ax = _shp_merge_tech_water.plot()

# %%
_shp_merge_tech_water.shape

# %%
_shp_merge_tech_forest = overlay(tech_high_med_dist,forest_med, how='intersection')

# %%
_shp_merge_tech_forest.columns

# %%
_shp_merge_tech_forest["geometry"].plot()

# %% [markdown]
# _shp_merge_tech_forest.shape

# %%
_shp_merge_water_forest = overlay(water_high_med_dist,forest_med, how='intersection')

# %%
_shp_merge_water_forest.shape

# %%
ax =_shp_district.plot(_shp_district,figsize=(5,5),color="none")
tech_high_med_dist.plot(color="brown",ax =ax,alpha = 0.5)
water_high_med_dist.plot(color ="blue",ax =ax,alpha = 0.5)
forest_med.plot(color ="green",ax =ax)
_shp_merge_tech_water.plot(color ="black",ax =ax)
_shp_merge_tech_forest.plot(color ="red",ax = ax)
# _shp_merge_water_forest.plot(color ="yellow", ax = ax)
plt.axis('off')
plt.savefig(get_in_workdir("_shp_merge_tech_water_forest.jpg"),dpi =1500)
plt.show()

# %%

# %%
