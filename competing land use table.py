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
# ## competing land use table(Distribution by size)
#

# %%
import geopandas as gpd
import pandas as pd
import numpy as np
from osgeo import gdal
import fiona


# %%
def get_rooted(stem):
    return "D:\\LiLa_Nagapattinam\\" + stem
def read_df_UT(stem):
    return gpd.read_file(get_rooted(stem)).to_crs(epsg = 4326)



# %%
lc_tech = read_df_UT('solar\\_rl_elev_rd_wat_co_trans_ar_sub_rdpx_trsub_tech\\LC_Solar_final_area_mask_1_Nagapattinam.shp')
shp_tech_high =read_df_UT("solar\\_rl_elev_rd_wat_co_trans_ar_sub_rdpx_trsub_trat_subat_rdat_ir_high\\LC_Solar_final_area_mask_1_Nagapattinam.shp")
shp_tech_med = read_df_UT("solar\\_rl_elev_rd_wat_co_trans_ar_sub_rdpx_trsub_trat_subat_rdat_ir_medatt\\LC_Solar_final_area_mask_1_Nagapattinam.shp")
_shp_district = read_df_UT("Practice\\Nagapattinam_proj32644.shp")
forest_med = read_df_UT("forest\\_ter_elev_watpot_ar_med\\LC_Forest_final_area_mask_1_Nagapattinam.shp")
shp_water_high =read_df_UT("water\\_wd_run_high\\LC_Water_final.shp")
shp_water_med =read_df_UT("water\\_wd_run_med\\LC_Water_final.shp")

# %%
# shp_tech_high=shp_tech_high.to_crs(epsg=4326)
# shp_tech_med=shp_tech_med.to_crs(epsg=4326)
# forest_med=forest_med.to_crs(epsg=4326)
# shp_water_high=shp_water_high.to_crs(epsg=4326)
# shp_water_med=shp_water_med.to_crs(epsg=4326)
# _shp_district=_shp_district.to_crs(epsg=4326)
# lc_tech = lc_tech.to_crs(4326)

# %%
water_high_med = gpd.pd.concat([shp_water_high,shp_water_med])
water_high_med_dist = gpd.overlay(_shp_district,water_high_med,how ="intersection")

forest_med = forest_med["geometry"]
forest_med = gpd.GeoDataFrame(forest_med)
water_high_med_dist = water_high_med_dist["geometry"]
water_high_med_dist = gpd.GeoDataFrame(water_high_med_dist)


# %%

# %%
crs_utm = 32644    
df = lc_tech
df = df.to_crs(crs_utm)


tag = 'water'
#olapbool - is there an overlap True or False
#'olap%'- What is teh overlap % in terms of area
#'olaparea'- what is the overlap area ?
#'origarea2'- What is teh area of original competing land use land polygon ?

df1 = pd.DataFrame(columns = ['olap%'+tag,'olaparea'+tag])
#if tehre are two polygons inetersting then store it as a list and those fields have to be declared as a object type column

df1['olap%'+tag]=df1['olap%'+tag].astype('object')
df1['olaparea'+tag]=df1['olaparea'+tag].astype('object')

#read teh second finalised  shapefile Type 2( Solar, Water or Forest)
finallands2 = water_high_med_dist
fdf2 = water_high_med_dist
fdf2=fdf2.to_crs(crs_utm)

#set spatial index for data for faster processing
sindex = fdf2.sindex

for i in range(len(df)):
    geometry = df.iloc[i]['geometry']
            
    fids = list(sindex.intersection(geometry.bounds))

    
    if fids:
       
       olaparea = ((fdf2.iloc[fids]['geometry'].intersection(geometry)).area).sum()
       olap_perc = olaparea*100/geometry.area
       olaparea = (olaparea/10**6)*247.1               
                   
    else:
       
       olaparea = 0
       olap_perc = 0
   
               
    df1.at[i,'olap%'+tag] =  olap_perc      
    df1.at[i,'olaparea'+tag] = olaparea


df = pd.concat([df,df1], axis= 1)


# %%
crs_utm = 32644    
df2 = lc_tech
df2=df2.to_crs(crs_utm)


tag = 'forest'
#olapbool - is there an overlap True or False
#'olap%'- What is teh overlap % in terms of area
#'olaparea'- what is the overlap area ?
#'origarea2'- What is teh area of original competing land use land polygon ?

df3 = pd.DataFrame(columns = ['olap%'+tag,'olaparea'+tag])
#if tehre are two polygons inetersting then store it as a list and those fields have to be declared as a object type column

df3['olap%'+tag]=df3['olap%'+tag].astype('object')
df3['olaparea'+tag]=df3['olaparea'+tag].astype('object')

#read teh second finalised  shapefile Type 2( Solar, Water or Forest)
finallands2 = forest_med
fdf2 = forest_med
fdf2=fdf2.to_crs(crs_utm)

#set spatial index for data for faster processing
sindex = fdf2.sindex

for i in range(len(df2)):
    geometry = df2.iloc[i]['geometry']
            
    fids = list(sindex.intersection(geometry.bounds))

    
    if fids:
       
       olaparea = ((fdf2.iloc[fids]['geometry'].intersection(geometry)).area).sum()
       olap_perc = olaparea*100/geometry.area
       olaparea = (olaparea/10**6)*247.1               
                   
    else:
       
       olaparea = 0
       olap_perc = 0
   
               
    df3.at[i,'olap%'+tag] =  olap_perc      
    df3.at[i,'olaparea'+tag] = olaparea


df = pd.concat([df,df3], axis= 1)


# %%

# %%
lc_tech_A = df[df["area_class"] == "A"] #5 to 20
lc_tech_B = df[df["area_class"] == "B"]#20 to 100
lc_tech_C = df[df["area_class"] == "C"]#greater than 100

# %%
lc_tech_A["olapareawater"].sum()

# %%
lc_tech_B["olapareawater"].sum()

# %%
lc_tech_C["olapareawater"].sum()

# %%
df4 =pd.concat([df1,df3], axis= 1)

# %%
df4= df4[df4['olapareaforest'] != 0]
df4

# %% [markdown]
# ## Top 15 tech lands

# %%
shp_tech_high = shp_tech_high.sort_values(by=["area_acres"],ascending = False)

# %%
shp_tech_med = shp_tech_med.sort_values(by=["area_acres"],ascending = False)

# %%
shp_tech_med = shp_tech_med[:15]

# %%
shp_tech_high_med = gpd.pd.concat([shp_tech_high,shp_tech_med])

# %%

# %%
shp_tech_high_med_top=shp_tech_high_med.drop_duplicates(subset ="geometry",keep ="first")

# %%
shp_tech_high_med_top = shp_tech_high_med_top.reset_index()

# %%

# %%
# #input finalised lands of Type 1 ( Solar, Water or Forest)
# # finallands1 = shp_tech_high_med
crs_utm = 32644    
df = shp_tech_high_med_top
df = df.to_crs(crs_utm)


tag = 'water'
#olapbool - is there an overlap True or False
#'olap%'- What is teh overlap % in terms of area
#'olaparea'- what is the overlap area ?
#'origarea2'- What is teh area of original competing land use land polygon ?

df1 = pd.DataFrame(columns = ['olap%'+tag,'olaparea'+tag])
#if tehre are two polygons inetersting then store it as a list and those fields have to be declared as a object type column

df1['olap%'+tag]=df1['olap%'+tag].astype('object')
df1['olaparea'+tag]=df1['olaparea'+tag].astype('object')

#read teh second finalised  shapefile Type 2( Solar, Water or Forest)

fdf2 = water_high_med_dist
fdf2=fdf2.to_crs(crs_utm)

#set spatial index for data for faster processing
sindex = fdf2.sindex

for i in range(len(df)):
    geometry = df.iloc[i]['geometry']
            
    fids = list(sindex.intersection(geometry.bounds))

    
    if fids:
       
       olaparea = ((fdf2.iloc[fids]['geometry'].intersection(geometry)).area).sum()
       olap_perc = olaparea*100/geometry.area
       olaparea = (olaparea/10**6)*247.1               
                   
    else:
       
       olaparea = 0
       olap_perc = 0
   
               
    df1.at[i,'olap%'+tag] =  olap_perc      
    df1.at[i,'olaparea'+tag] = olaparea


df = pd.concat([df,df1], axis= 1)


# %%

# %%
# #input finalised lands of Type 1 ( Solar, Water or Forest)
# # finallands1 = shp_tech_high_med
crs_utm = 32644    
df2 = shp_tech_high_med_top
df2=df2.to_crs(crs_utm)


tag = 'forest'
#olapbool - is there an overlap True or False
#'olap%'- What is teh overlap % in terms of area
#'olaparea'- what is the overlap area ?
#'origarea2'- What is teh area of original competing land use land polygon ?

df3 = pd.DataFrame(columns = ['olap%'+tag,'olaparea'+tag])
#if tehre are two polygons inetersting then store it as a list and those fields have to be declared as a object type column

df3['olap%'+tag]=df3['olap%'+tag].astype('object')
df3['olaparea'+tag]=df3['olaparea'+tag].astype('object')

#read teh second finalised  shapefile Type 2( Solar, Water or Forest)

fdf2 = forest_med
fdf2=fdf2.to_crs(crs_utm)

#set spatial index for data for faster processing
sindex = fdf2.sindex

for i in range(len(df2)):
    geometry = df2.iloc[i]['geometry']
            
    fids = list(sindex.intersection(geometry.bounds))

    
    if fids:
       
       olaparea = ((fdf2.iloc[fids]['geometry'].intersection(geometry)).area).sum()
       olap_perc = olaparea*100/geometry.area
       olaparea = (olaparea/10**6)*247.1               
                   
    else:
       
       olaparea = 0
       olap_perc = 0
   
               
    df3.at[i,'olap%'+tag] =  olap_perc      
    df3.at[i,'olaparea'+tag] = olaparea


df = pd.concat([df,df3], axis= 1)


# %%
df = df
input_raster ="D:\\LiLa_Nagapattinam\\Supporting_info\\DEM_T44PLT_proj32644_filled_slope.tif"
df.geometry = df.geometry.buffer(0)  
outputdf = pd.DataFrame()
for i in range(len(df)):
        input_shp =  'D:\\LiLa_Nagapattinam\\workdir\\temp.shp'

        #Each basin geometry converted to shapefile
        selection = df['geometry'][i:i+1]
        #selection = bdf['geometry'][i:i+1]
        selection.to_file(input_shp)

        output_raster = 'D:\\LiLa_Nagapattinam\\workdir\\temp.tif'

        ds = gdal.Warp(output_raster,
                      input_raster,
                      format = 'GTiff',
                      cutlineDSName = input_shp,
                      cropToCutline=True,
                      )
        ds = None


        raster = gdal.Open(output_raster, gdal.GA_ReadOnly)
        rasterarr = raster.ReadAsArray()
        #remove nodata values
        rasterarr = rasterarr[rasterarr!= -9999]

        if (np.size(rasterarr)==0):
            outputdf.at[i, 'min']=np.nan
            outputdf.at[i , 'max']=np.nan
            outputdf.at[i , 'mean']=np.nan
            outputdf.at[i , '25percentile']=np.nan
            outputdf.at[i , '75percentile']=np.nan

        else:    

            outputdf.at[i, 'min']=rasterarr.min()
            outputdf.at[i , 'max']=rasterarr.max()
            outputdf.at[i , 'mean']=rasterarr.mean()
            outputdf.at[i , '25percentile']=np.percentile(rasterarr,25)
            outputdf.at[i , '75percentile']=np.percentile(rasterarr,75)

# %%
df = pd.concat([df,outputdf], axis= 1)

# %%
df

# %%
