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


# %% [markdown]
# ## Importing the village dataset and Technical potential(forest) dataset

# %%
village = read_df_UT("forest\\output_forest_stats_nobuffer.shp")
lc_tech = read_df_UT('forest\\_ter_ar_tech\\LC_Forest_final_area_mask_1_Nagapattinam.shp')

# %%
village.columns

# %% [markdown]
# ## Simple visualization

# %%
# %matplotlib inline
ax = village.plot(column='fcover%', scheme='QUANTILES', k=4, \
             cmap='viridis', legend=True,
             legend_kwds={'loc': 'center left', 'bbox_to_anchor':(1,0.5)})

# %%
# ax = village.plot(column='fper1000', scheme='QUANTILES', k=4, \
#              cmap='viridis', legend=True,
#              legend_kwds={'loc': 'center left', 'bbox_to_anchor':(1,0.5)})

# %%
village.NAME.nunique()

# %%
village.NAME.value_counts()

# %% [markdown]
# ### Cross checking the technical potential

# %%
# Pudupattinam = village[village["NAME"] == 'Pudupattinam']

# %%
# Pudupattinam.to_file("D:\\LiLa_Nagapattinam\\workdir\\Pudupattinam.shp")

# %%
# Pudupattinam.plot(cmap="viridis")

# %%
print(village.columns.tolist())


# %%

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
            count = (fdf2.iloc[fids]['geometry'].intersection(geometry)).count()
            olap_perc = olaparea*100/geometry.area
            olaparea = (olaparea/10**6)*247.1               
        else:
            olaparea = 0
            olap_perc = 0
        df1.at[i,'olap%'+tag] =  olap_perc      
        df1.at[i,'olaparea'+tag] = olaparea
        df1.at[i,'count'+tag] = count
    return pd.concat([df,df1], axis= 1)


# %%
village_tech = find_overlap_area(village,"tech",lc_tech)

# %%

# %%
village_tech["totar_acres"] = ((village_tech.geometry.area)/10**6) * 247.105
village_tech["f_totar_acres"] = village_tech["f_totarkm2"] * 247.105
village_tech["tree_cover_perc"] = (village_tech["f_totar_acres"] / village_tech["totar_acres"] ) * 100
village_tech["f_cover_1000"] = (village_tech["f_totar_acres"] / village_tech["totpop"])*1000
village_tech["total_potential"] = (village_tech["f_totar_acres"] + village_tech["olapareatech"])/ village_tech["totar_acres"]
village_tech["total_potential"] = village_tech["total_potential"]*100

# %%
village_tech["f_totar_acres"].min()
village_tech["f_totar_acres"] = village_tech["f_totar_acres"].replace(0,np.nan)

# %%
village_tech["ratio"]  =village_tech["olapareatech"] / village_tech["f_totar_acres"]
village_tech["Potential_avail"] = village_tech["olapareatech"]/(village_tech["f_totar_acres"]+village_tech["olapareatech"])
village_tech["Potential_avail"] = village_tech["Potential_avail"] *100
print(village_tech.columns.to_list())

# %%
village_tech = village_tech.to_crs(4326)

# %%
village_tech['coords'] = village_tech['geometry'].apply(lambda x: x.representative_point().coords[:])
village_tech['coords'] = [coords[0] for coords in village_tech['coords']]
village_tech["coords"].tolist()
village_tech[['lat', 'lon']] = gpd.GeoDataFrame(village_tech['coords'].tolist(), index=village_tech.index)

# %%
village_tech = village_tech[["NAME","lat","lon","totpop","totar_acres","f_totar_acres","olapareatech","counttech","tree_cover_perc","f_cover_1000","total_potential","Potential_avail","ratio"]]

# %%
# village_tech.to_excel("D:\\LiLa_Nagapattinam\\village_overall_data.xlsx")
