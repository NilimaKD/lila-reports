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
# ## Competing land use table(Distribution by size)
#

# %%
import geopandas as gpd
import pandas as pd
import numpy as np
from osgeo import gdal


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

# %%
water_high_med = gpd.pd.concat([shp_water_high,shp_water_med])
water_high_med_dist = gpd.overlay(_shp_district,water_high_med,how ="intersection")

forest_med = forest_med["geometry"]
forest_med = gpd.GeoDataFrame(forest_med)
water_high_med_dist = water_high_med_dist["geometry"]
water_high_med_dist = gpd.GeoDataFrame(water_high_med_dist)


# %%
lc_tech.shape


# %%
def find_overlap_area(df,tag,fdf2):
    crs_utm = 32644    
    df = df.to_crs(crs_utm)
    df1 = pd.DataFrame(columns = ['olap%'+tag,'olaparea'+tag])
    df1['olap%'+tag]=df1['olap%'+tag].astype('object')
    df1['olaparea'+tag]=df1['olaparea'+tag].astype('object')
    
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
    return pd.concat([df,df1], axis= 1)


# %%
df_water = find_overlap_area(lc_tech,"water",water_high_med_dist)



# %%
df_water_forest = find_overlap_area(df_water,"forest",forest_med)

# %%

# %%
lc_tech_A = df_water_forest[df_water_forest["area_class"] == "A"] #5 to 20
lc_tech_B = df_water_forest[df_water_forest["area_class"] == "B"]#20 to 100
lc_tech_C = df_water_forest[df_water_forest["area_class"] == "C"]#greater than 100

# %%
lc_tech_A["olapareawater"].sum()

# %%
lc_tech_B["olapareawater"].sum()

# %%
lc_tech_C["olapareawater"].sum()

# %%
# df4 =pd.concat([df1,df3], axis= 1)

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
df_water_top = find_overlap_area(shp_tech_high_med_top,"water",water_high_med_dist)

# %%
df_water_forest_top = find_overlap_area(df_water_top,"forest",forest_med)

# %%
df_water_forest_top

# %%
df = df_water_forest_top
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
# df.to_csv("D:\\LiLa_Nagapattinam\\workdir\\top15_lands.csv")

# %%
df

# %%
