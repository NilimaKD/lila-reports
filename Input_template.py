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
# ### define func for creating Area_acres and area_class

# %% [markdown]
# #### change the values of range according to the critreria

# %%
def area_acres(df):
    crs_utm = 32644 
    df = df.to_crs(crs_utm)
    df["area_acres"] = (((df.geometry.area)/10**6)*247.105)
    a = df["area_acres"].max()
    def area_class(df):
        if 5<= df["area_acres"] < 20:
            return "A"
        elif 20<= df["area_acres"] < 100:
            return "B"
        elif 100<= df["area_acres"] <= a:
            return "C"  
        else:
            return "D"
    df["area_class"] =df.apply(area_class, axis=1)
    return df[["area_acres","area_class","geometry"]]


# %% [markdown]
# ### define func for creating Area_hect and area_class

# %%
def area_hect(df):
    crs_utm = 32644 
    df = df.to_crs(crs_utm)
    df["area_hect"] = ((df.geometry.area)/10**4)
    a = df["area_hect"].max()
    def area_class(df):
        if 5 <= df["area_hect"] < 20:
            return "A"
        elif 20<= df["area_hect"] < 100:
            return "B"
        elif 100<= df["area_hect"] <= a:
            return "C"  
        else:
            return "D"
    df["area_class"] =df.apply(area_class, axis=1)
    return df[["area_hect","area_class","geometry"]]
