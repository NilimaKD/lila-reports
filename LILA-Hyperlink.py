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
import pandas as pd
import folium
from osgeo import gdal
from folium import plugins,Vega
from folium import raster_layers
#from matplotlib import cm
import numpy as np
import branca.colormap as cm
#import branca
import matplotlib
from matplotlib import colors
import matplotlib.pyplot as plt
from folium.plugins import MarkerCluster,FloatImage, Draw, MeasureControl
# import osmnx as ox
# import geopandas
# import vincent
# from vincent import Bar,AxisProperties, PropertySet, ValueRef
import geopandas as gpd
import os
# import plotly.graph_objects as go
# import branca
# import plotly.io as pio
# from plotly.subplots import make_subplots
from matplotlib import colors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


# %%
def get_rooted(stem):
    return "D:\\LiLa_Nagapattinam\\" + stem
def read_df_UT(stem):
    return gpd.read_file(get_rooted(stem)).to_crs(epsg = 4326)


# %%

# %%
roads_prim_sec = read_df_UT("workdir\\hyperlink_file\\_shp_roads_primary_secondary.shp")
def regioncolors(roads_prim_sec):
    if roads_prim_sec['highway'] == 'primary':
        return '#df2a29'
    elif roads_prim_sec['highway']  == 'secondary':
        return '#f0b8b3'
    
roads_prim_sec["color"] = roads_prim_sec.apply(regioncolors, axis=1)
roads_prim_sec.to_file('roads_prim_sec.json', driver='GeoJSON')
roads_prim_sec = os.path.join('', 'roads_prim_sec.json')



substations =gpd.read_file("D:\\LiLa_Nagapattinam\\workdir\\hyperlink_file\\Untitled_layer-point.shp")
substations.to_file('substations.json', driver='GeoJSON')
substations = os.path.join('', 'substations.json')



powerlines = read_df_UT("workdir\\hyperlink_file\\powerlines.shp")
powerlines.to_file('powerlines.json', driver='GeoJSON')
powerlines = os.path.join('', 'powerlines.json')

_shp_district = read_df_UT("Practice\\Nagapattinam_proj32644.shp")
_shp_district.to_file('_shp_district.json', driver='GeoJSON')
_shp_district = os.path.join('', '_shp_district.json')

GHI = 'D:\\LiLa_Nagapattinam\\workdir\\hyperlink_file\\GHIepsg_Nagapattinam.tif'
driver=gdal.GetDriverByName('GTiff')
driver.Register() 
ds_ghi = gdal.Open(GHI)
if ds_ghi is None:
    print('Could not open')

#Raster convert to array in numpy
band_ghi=ds_ghi.GetRasterBand(1)

data_ghi= band_ghi.ReadAsArray()


#Get coordinates, cols and rows
geotransform = ds_ghi.GetGeoTransform()
cols_ghi = ds_ghi.RasterXSize
rows_ghi = ds_ghi.RasterYSize

#Get extent
xmin_ghi=geotransform[0]
ymax_ghi=geotransform[3]
xmax_ghi=xmin_ghi+cols_ghi*geotransform[1]
ymin_ghi=ymax_ghi+rows_ghi*geotransform[5]

centerx=(xmin_ghi+xmax_ghi)/2
centery=(ymin_ghi+ymax_ghi)/2


def get_color_ghi(x):
   
    if 1800<= x <= 1879.16:
        return (233,255,112,1)
    elif 1879.16 < x <=1958.33 :
        return (246,222,38,1)
    elif 1958.33 < x <=1978 :
        return (251,179,28,1)
    elif 1978 < x <=1997.67 :
        return (255,136,17,1)
    elif 1997.67 < x <=2017.34 :
        return (253,109,19,1)
    elif 2017.34  < x <=2037.58 :
        return (251,81,20,1)
    elif 2037.58 < x <=2275 :
        return (167,11,11,1)

    else:
        return (255,0,0,0)

    
outfile = 'D:\\LiLa_Nagapattinam\\workdir\\hyperlink_file\\slope_dstepsg.tif'
#Open raster file
driver=gdal.GetDriverByName('GTiff')
driver.Register() 
ds1 = gdal.Open(outfile)
if ds1 is None:
    print('Could not open')

#Raster convert to array in numpy
band1=ds1.GetRasterBand(1)

datasetr1= band1.ReadAsArray()


#Get coordinates, cols and rows
geotransform = ds1.GetGeoTransform()
cols2 = ds1.RasterXSize
rows2 = ds1.RasterYSize

#Get extent
xmin2=geotransform[0]
ymax2=geotransform[3]
xmax2=xmin2+cols2*geotransform[1]
ymin2=ymax2+rows2*geotransform[5]

#Get Central point
centerx=(xmin2+xmax2)/2
centery=(ymin2+ymax2)/2

data_array = ds1.ReadAsArray().astype(dtype=float)
# print(data_array)


if np.any(data_array < 0):
    data_array[data_array < 0 ] = np.nan
    

AVC_grey = LinearSegmentedColormap.from_list('testCmap1', colors=['#F0F0F0', '#c3c3c3', '#969696', '#686868'], N=256)

    
m= folium.Map(location=[11.18, 79.7071], zoom_start=10)

tile = folium.TileLayer(
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr = 'Esri',
        name = 'Esri Satellite',
        overlay = False,
        control = True
       ).add_to(m)




loc = 'District features'
title_html = '''
             <h3 align="center" style="font-size:16px"><b>{}</b></h3>
             '''.format(loc)   


m.get_root().html.add_child(folium.Element(title_html))


m.add_child(raster_layers.ImageOverlay(
    image=data_ghi,
    show=False,
    name = 'GHI',
    bounds=[[ymin_ghi, xmin_ghi], [ymax_ghi, xmax_ghi]],
    colormap=lambda x: get_color_ghi(x),
    interactive=True,
    mercator_project = True,
    opacity=0.8
))#.add_to(m)







folium.GeoJson(
    _shp_district,
     name='dist_boundary',
    show = True,
    control = False,
        style_function=lambda feature: {
        'fillColor': "none",
        'color' : "black",
        'weight' : 1,
        'fillOpacity' : 0.5,
        },
#          highlight_function=lambda x: {'weight':5,'color':'yellow'},
#          tooltip=folium.features.GeoJsonTooltip(fields=['AREA'],labels=False, toLocaleString=True)
      #smooth_factor=2.0
           
    ).add_to(m)



folium.GeoJson(
    powerlines,
     name='Powerlines',
    show = False,
        style_function=lambda feature: {
        'fillColor':"#C4171C ",
        'color' : "#C4171C",
        'weight' : 1,
        'fillOpacity' : 0.5,
        },
         highlight_function=lambda x: {'weight':5,'color':'yellow'},
         tooltip=folium.features.GeoJsonTooltip(fields=['voltage'],aliases=["Voltage :"],labels=True, toLocaleString=True)
      #smooth_factor=2.0
           
    ).add_to(m)



m.add_child(raster_layers.ImageOverlay(
    image=data_array,
    show=False,
    name = 'Slope',
    bounds=[[ymin2, xmin2], [ymax2, xmax2]],
    colormap=AVC_grey,
    interactive=True,
    mercator_project = True,
    opacity=0.5
))#.add_to(m)


folium.GeoJson(
    substations,
     name='Substations',
    show = False,
        style_function=lambda feature: {
        'fillColor':"#dd2c0e",
        'color' : "#dd2c0e",
        'weight' : 1,
        'fillOpacity' : 0.5,
        },
         highlight_function=lambda x: {'weight':5,'color':'yellow'},
         tooltip=folium.features.GeoJsonTooltip(fields=['Name'],aliases=["Capacity(KW):"],labels=True, toLocaleString=True)
      #smooth_factor=2.0
           
    ).add_to(m)




folium.GeoJson(
    roads_prim_sec,
     name='Roads',
     show = False,
         style_function=lambda feature: {
         'fillColor': feature['properties']['color'],
         'color' : feature['properties']['color'],
         'weight' : 1,
         'fillOpacity' : 0.5,
         },
         highlight_function=lambda x: {'weight':5,'color':'yellow'},
         tooltip=folium.features.GeoJsonTooltip(fields=['highway'],aliases=["Type :"] ,labels=True, toLocaleString=True)
      #smooth_factor=2.0
           
    ).add_to(m)


plugins.Fullscreen(
    position="topright",
    title="Expand me",
    title_cancel="Exit me",
    force_separate_button=True,
).add_to(m)

minimap = plugins.MiniMap()
m.add_child(minimap)



# Add drawinf controls
draw = Draw()
draw.add_to(m)
m.add_child(MeasureControl())


folium.LayerControl().add_to(m)
#plugins.Geocoder().add_to(m)
m.save('D:\\LiLa_Nagapattinam\\Supporting_info\\Features.html')

# %%

_shp_district = read_df_UT("Practice\\Nagapattinam_proj32644.shp")
_shp_district.to_file('_shp_district.json', driver='GeoJSON')
_shp_district = os.path.join('', '_shp_district.json')


filename = 'D:\\LiLa_Nagapattinam\\workdir\\hyperlink_file\\lc_dstepsg.tif'


#Open raster file
driver=gdal.GetDriverByName('GTiff')
driver.Register() 
ds = gdal.Open(filename) 
if ds is None:
    print('Could not open')

#Get coordinates, cols and rows
geotransform = ds.GetGeoTransform()
cols = ds.RasterXSize 
rows = ds.RasterYSize 

#Get extent
xmin=geotransform[0]
ymax=geotransform[3]
xmax=xmin+cols*geotransform[1]
ymin=ymax+rows*geotransform[5]

#Get Central point
centerx=(xmin+xmax)/2
centery=(ymin+ymax)/2

#Raster convert to array in numpy
bands = ds.RasterCount
band=ds.GetRasterBand(1)
dataset= band.ReadAsArray(0,0,cols,rows)
dataimage=dataset

color_labels = np.unique(dataset)
print(color_labels)

#dataimage1=np.where(dataset==100, 7, dataimage1) 

# extract clouds
#dataimage1=np.where(dataset==100, 1, 0)
# extract unclassified
dataimage2=np.where(dataset==0, 100, 0) 
#extract Barren
dataimage3=np.where(dataset==1, 1, 0) 
#extract Sparse Vegetation
dataimage4=np.where(dataset==2, 2, 0)
#extract CropLand
dataimage5=np.where(dataset==3, 3, 0)
#extract Forests
dataimage6=np.where(dataset==4, 4, 0)
#extract Water
dataimage7=np.where(dataset==5, 5, 0)  
#extract Urban
dataimage8=np.where(dataset==6, 6, 0) 


dataimage9 = np.where(dataset==50, 5, 0)





m= folium.Map(location=[centery, centerx], zoom_start=10)    

g2 = folium.plugins.FeatureGroupSubGroup(m, 'Unused',show=False)
g3 = folium.plugins.FeatureGroupSubGroup(m, 'Sparse vegetation',show=False)
g4 = folium.plugins.FeatureGroupSubGroup(m, 'Cropland',show=False)
g5 = folium.plugins.FeatureGroupSubGroup(m, 'Forest',show=False)
g6 = folium.plugins.FeatureGroupSubGroup(m, 'Water',show=False)
g7 = folium.plugins.FeatureGroupSubGroup(m, 'Built-up',show=False)


m.add_child(g2)
m.add_child(g3)
m.add_child(g4)
m.add_child(g5)
m.add_child(g6)
m.add_child(g7)

tile = folium.TileLayer(
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr = 'Esri',
        name = 'Esri Satellite',
        overlay = False,
        control = True
       ).add_to(m)




loc = 'Land cover map'
title_html = '''
             <h3 align="center" style="font-size:16px"><b>{}</b></h3>
             '''.format(loc)   


m.get_root().html.add_child(folium.Element(title_html))



folium.GeoJson(
    _shp_district,
     name='dist_boundary',
    show = True,
    control = False,
        style_function=lambda feature: {
        'fillColor': "none",
        'color' : "black",
        'weight' : 1,
        'fillOpacity' : 0.5,
        },
#          highlight_function=lambda x: {'weight':5,'color':'yellow'},
#          tooltip=folium.features.GeoJsonTooltip(fields=['AREA'],labels=False, toLocaleString=True)
      #smooth_factor=2.0           
).add_to(m)



g2.add_child(raster_layers.ImageOverlay(
    image=dataimage3,
    show=False,
    name = 'Unused',
    bounds=[[ymin, xmin], [ymax, xmax]],
    colormap=lambda x: (1-x,1-x, 1-x,0+x*255),#R,G,B,red: (255,0,0)
    interactive=True,
    mercator_project = True,
    opacity=1
))#.add_to(m)

g3.add_child(raster_layers.ImageOverlay(
    image=dataimage4,
    show=False,
    name = 'Sparse Vegetation',
    bounds=[[ymin, xmin], [ymax, xmax]],
    colormap=lambda x: (253+x,0,253+x,0+x*127),#R,G,B,alpha Yellow: (255,255,0) (255,0,255)
    interactive=True,
    mercator_project = True,
    opacity=1
))#.add_to(m)

g4.add_child(raster_layers.ImageOverlay(
    image=dataimage5,
    show=False,
    name = 'CropLand',
    bounds=[[ymin, xmin], [ymax, xmax]],
    colormap=lambda x: (3-x,13-x, 3-x,0+x*85),#R,G,B,alpha Lime Green (0,255,0)(0,10,0)
    interactive=True,
    mercator_project = True,
    opacity=1
))#.add_to(m)

g5.add_child(raster_layers.ImageOverlay(
    image=dataimage6,
    show=False,
    name = 'Forest',
    bounds=[[ymin, xmin], [ymax, xmax]],
    colormap=lambda x: (4-x,251+x,4-x,0+x*63),#R,G,B,alpha Dark Green (0,10,0)(0,255,0)
    interactive=True,
    mercator_project = True,
    opacity=1
))#.add_to(m)

g6.add_child(raster_layers.ImageOverlay(
    image=dataimage7,
    show=False,
    name = 'Water',
    bounds=[[ymin, xmin], [ymax, xmax]],
    colormap=lambda x: (5-x,250+x, 250+x,0+x*51),#R,G,B,alpha Blue (0, 255, 255)
    interactive=True,
    mercator_project = True,
    opacity=1
))#.add_to(m)

g7.add_child(raster_layers.ImageOverlay(
    image=dataimage8,
    show=False,
    name = 'Urban',
    bounds=[[ymin, xmin], [ymax, xmax]],
    colormap=lambda x: (249+x,6-x,6-x,0+x*42),#R,G,B,alpha Fuchsia (255,0,255)(255,0,0)
    interactive=True,
    mercator_project = True,
    opacity=1
))#.add_to(m)

g6.add_child(raster_layers.ImageOverlay(
    image=dataimage9,
    show=False,
    name = 'Water bodies',
    bounds=[[ymin, xmin], [ymax, xmax]],
    colormap=lambda x: (5-x,250+x, 250+x,0+x*51),#R,G,B,alpha Blue (0, 255, 255)
    interactive=True,
    mercator_project = True,
    opacity=1
))#.add_to(m)


plugins.Fullscreen(
    position="topright",
    title="Expand me",
    title_cancel="Exit me",
    force_separate_button=True,
).add_to(m)

minimap = plugins.MiniMap()
m.add_child(minimap)



# Add drawinf controls
draw = Draw()
draw.add_to(m)
m.add_child(MeasureControl())


folium.LayerControl().add_to(m)
#plugins.Geocoder().add_to(m)
m.save('D:\\LiLa_Nagapattinam\\Supporting_info\\Landcover.html')

# %%

# %%

# %%

# %%


# %%
_shp_merge_tech_water = read_df_UT("workdir\\hyperlink_file\\_shp_merge_tech_water.shp")
_shp_merge_tech_water = _shp_merge_tech_water.round({"area_acres":0})
_shp_merge_tech_water.to_file('_shp_merge_tech_water.json', driver='GeoJSON')
_shp_merge_tech_water = os.path.join('', '_shp_merge_tech_water.json')


_shp_merge_tech_forest = read_df_UT("workdir\\hyperlink_file\\_shp_merge_tech_forest.shp")
_shp_merge_tech_forest = _shp_merge_tech_forest.round({"area_acres":0})
_shp_merge_tech_forest.to_file('_shp_merge_tech_forest.json', driver='GeoJSON')
_shp_merge_tech_forest = os.path.join('', '_shp_merge_tech_forest.json')



lc_tech = read_df_UT('solar\\_rl_elev_rd_wat_co_trans_ar_sub_rdpx_trsub_tech\\LC_Solar_final_area_mask_1_Nagapattinam.shp')
lc_tech = lc_tech.round({"area_acres":0})
lc_tech.to_file('lc_tech.json', driver='GeoJSON')
lc_tech = os.path.join('', 'lc_tech.json')



_shp_district = read_df_UT("Practice\\Nagapattinam_proj32644.shp")
_shp_district.to_file('_shp_district.json', driver='GeoJSON')
_shp_district = os.path.join('', '_shp_district.json')

m= folium.Map(location=[11.18, 79.7071], zoom_start=10)




loc = 'Competing land-use for climate action'
title_html = '''
             <h3 align="center" style="font-size:16px"><b>{}</b></h3>
             '''.format(loc)   


m.get_root().html.add_child(folium.Element(title_html))




tile = folium.TileLayer(
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr = 'Esri',
        name = 'Esri Satellite',
        overlay = False,
        control = True
       ).add_to(m)



# folium.GeoJson(
#     _shp_merge_tech_water_forest,
#      name='Forest + Water',
#     show = False,
#         style_function=lambda feature: {
#         'fillColor': " #e73429",
#         'color' : "#e73429",
#         'weight' : 3,
#         'fillOpacity' : 0.5,
#         },
# #          highlight_function=lambda x: {'weight':5,'color':'yellow'},
# #          tooltip=folium.features.GeoJsonTooltip(fields=['DN'],labels=False, toLocaleString=True)
#       #smooth_factor=2.0
           
#     ).add_to(m)

folium.GeoJson(
    lc_tech,
     name='Solar Technical Potential',
    show = False,
        style_function=lambda feature: {
        'fillColor': "#FBB034",
        'color' : "#FBB034",
        'weight' : 3,
        'fillOpacity' : 0.5,
        },
         highlight_function=lambda x: {'weight':5,'color':'yellow'},
         tooltip=folium.features.GeoJsonTooltip(fields=['area_acres'],aliases=["Area (acres) :"],labels=True, toLocaleString=True)
      #smooth_factor=2.0
           
    ).add_to(m)


folium.GeoJson(
    _shp_merge_tech_forest,
     name='Forest',
    show = False,
        style_function=lambda feature: {
        'fillColor': "#B1D355",
        'color' : "#B1D355",
        'weight' : 3,
        'fillOpacity' : 0.5,
        },
         highlight_function=lambda x: {'weight':5,'color':'yellow'},
         tooltip=folium.features.GeoJsonTooltip(fields=['area_acres'],aliases=["Area (acres) :"],labels=True, toLocaleString=True)
      #smooth_factor=2.0
           
    ).add_to(m)


folium.GeoJson(
    _shp_merge_tech_water,
     name='Water',
    show = False,
        style_function=lambda feature: {
        'fillColor': "#00B9F2",
        'color' : "#00B9F2",
        'weight' : 3,
        'fillOpacity' : 0.5,
        },
         highlight_function=lambda x: {'weight':5,'color':'yellow'},
         tooltip=folium.features.GeoJsonTooltip(fields=['area_acres'],aliases=["Area (acres) :"],labels=True, toLocaleString=True)
      #smooth_factor=2.0
           
    ).add_to(m)



folium.GeoJson(
    _shp_district,
     name='dist_boundary',
    show = False,
    control = False,
        style_function=lambda feature: {
        'fillColor': "none",
        'color' : "black",
        'weight' : 1,
        'fillOpacity' : 0.5,
        },
#          highlight_function=lambda x: {'weight':5,'color':'yellow'},
#          tooltip=folium.features.GeoJsonTooltip(fields=['AREA'],labels=False, toLocaleString=True)
      #smooth_factor=2.0
           
    ).add_to(m)

plugins.Fullscreen(
    position="topright",
    title="Expand me",
    title_cancel="Exit me",
    force_separate_button=True,
).add_to(m)

minimap = plugins.MiniMap()
m.add_child(minimap)



# Add drawinf controls
draw = Draw()
draw.add_to(m)
m.add_child(MeasureControl())


folium.LayerControl().add_to(m)

m.save('D:\\LiLa_Nagapattinam\\Supporting_info\\Competinguse.html')


# %%

# %%

# %%
_shp_district = read_df_UT("Practice\\Nagapattinam_proj32644.shp")
_shp_district.to_file('_shp_district.json', driver='GeoJSON')
_shp_district = os.path.join('', '_shp_district.json')

lc_tech = read_df_UT("workdir\\hyperlink_file\\lc_tech.shp")

def regioncolors(lc_tech):
    if lc_tech['area_class'] == 'A':
        return '#646464'
    elif lc_tech['area_class']  == 'B':
        return '#BDA383'
    elif lc_tech['area_class']  == 'C':
        return '#FBAF30'
lc_tech["color"] = lc_tech.apply(regioncolors, axis=1)
lc_tech = lc_tech.round({"area_acres":0})
lc_tech["area_class"] = lc_tech["area_class"].replace(['A','B','C'],['Small area','Medium area','Large area'])
lc_tech.to_file('lc_tech.json', driver='GeoJSON')
lc_tech = os.path.join('', 'lc_tech.json')



No_potential = read_df_UT("workdir\\hyperlink_file\\No_potential.shp")
No_potential["Area_acres"] = No_potential["Area_acres"].replace([0],['< 1'])
No_potential.to_file('No_potential.json', driver='GeoJSON')
No_potential = os.path.join('', 'No_potential.json')



theo_only = read_df_UT("workdir\\hyperlink_file\\theo_only.shp")
theo_only = theo_only.round({"area_acres":0})
theo_only["area_acres"] = theo_only["area_acres"].replace([0],["< 1"])
theo_only.to_file('theo_only.json', driver='GeoJSON')
theo_only = os.path.join('', 'theo_only.json')


med = read_df_UT("workdir\\hyperlink_file\\med.shp")
med = med[["geometry","area_acres"]]
med = med.round({"area_acres":0})
med = gpd.GeoDataFrame(med)
# med = med.to_crs(4326)
# med.head()
med.to_file('med.json', driver='GeoJSON')
med = os.path.join('', 'med.json')



low = read_df_UT("workdir\\hyperlink_file\\low.shp")
low = low.round({"area_acres":0})
low.to_file('low.json', driver='GeoJSON')
low = os.path.join('', 'low.json')



high = read_df_UT("solar\\_rl_elev_rd_wat_co_trans_ar_sub_rdpx_trsub_trat_subat_rdat_ir_high\\LC_Solar_final_area_mask_1_Nagapattinam.shp")
high = high.round({"area_acres":0})
high.to_file('high.json', driver='GeoJSON')
high = os.path.join('', 'high.json')


m= folium.Map(location=[11.18, 79.7071], zoom_start=10)


tile = folium.TileLayer(
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr = 'Esri',
        name = 'Esri Satellite',
        overlay = False,
        control = True
       ).add_to(m)




loc = 'Land potential and distribution by size'
title_html = '''
             <h3 align="center" style="font-size:16px"><b>{}</b></h3>
             '''.format(loc)   


m.get_root().html.add_child(folium.Element(title_html))



folium.GeoJson(
    _shp_district,
    name='dist_boundary',
    show = False,
    control = False,
        style_function=lambda feature: {
        'fillColor': "none",
        'color' : "black",
        'weight' : 1,
        'fillOpacity' : 0.5,
        },
#          highlight_function=lambda x: {'weight':5,'color':'yellow'},
#          tooltip=folium.features.GeoJsonTooltip(fields=['AREA'],labels=False, toLocaleString=True)
      #smooth_factor=2.0
           
    ).add_to(m)


folium.GeoJson(
    No_potential,
    name='No potential',
    show = False,
        style_function=lambda feature: {
        'fillColor': "#424242",
        'color' : "#424242",
        'weight' : 3,
        'fillOpacity' : 0.5,
        },
         highlight_function=lambda x: {'weight':5,'color':'yellow'},
         tooltip=folium.features.GeoJsonTooltip(fields=['Area_acres'],aliases =["Area (acres) :"],labels=True, toLocaleString=True),
      smooth_factor=2.0
           
    ).add_to(m)


folium.GeoJson(
    theo_only,
    name='Theoritical potential',
    show = False,
        style_function=lambda feature: {
        'fillColor':"#997D41",
        'color' : "#997D41",
        'weight' : 3,
        'fillOpacity' : 0.5,
        },
         highlight_function=lambda x: {'weight':5,'color':'yellow'},
         tooltip=folium.features.GeoJsonTooltip(fields=['area_acres'],aliases =["Area (acres) :"],labels=True, toLocaleString=True),
      smooth_factor=2.0
           
    ).add_to(m)



folium.GeoJson(
    lc_tech,
    name='Technical potential',
    show = False,
        style_function=lambda feature: {
        'fillColor':feature['properties']['color'],
        'color' : feature['properties']['color'],
        'weight' : 3,
        'fillOpacity' : 0.5,
        },
         highlight_function=lambda x: {'weight':5,'color':'yellow'},
         tooltip=folium.features.GeoJsonTooltip(fields=['area_acres','area_class'],aliases =["Area (acres) :","Class :"],labels=True, toLocaleString=True),
      smooth_factor=2.0
           
    ).add_to(m)



folium.GeoJson(
    low,
    name='Low technical potential',
    show = False,
        style_function=lambda feature: {
        'fillColor': "#5A3228",
        'color' : "#5A3228",
        'weight' : 3,
        'fillOpacity' : 0.5,
        },
         highlight_function=lambda x: {'weight':5,'color':'yellow'},
         tooltip=folium.features.GeoJsonTooltip(fields=['area_acres'],aliases =["Area (acres) :"],labels=True, toLocaleString=True),
      smooth_factor=2.0
           
    ).add_to(m)


folium.GeoJson(
     med,
     name='Medium technical potential',
     show = False,
         style_function=lambda feature: {
         'fillColor': "#A77145",
         'color' : "#A77145",
         'weight' : 3,
         'fillOpacity' : 0.5,
         },
         highlight_function=lambda x: {'weight':5,'color':'yellow'},
         tooltip=folium.features.GeoJsonTooltip(fields=['area_acres'],aliases =["Area (acres) :"],labels=True, toLocaleString=True),
      smooth_factor=2.0
           
    ).add_to(m)



folium.GeoJson(
    high,
    name='High technical potential',
    show = False,
        style_function=lambda feature: {
        'fillColor': "#FBAF31",
        'color' : "#FBAF31",
        'weight' : 3,
        'fillOpacity' : 0.5,
        },
         highlight_function=lambda x: {'weight':5,'color':'yellow'},
         tooltip=folium.features.GeoJsonTooltip(fields=['area_acres'],aliases =["Area (acres) :"],labels=True, toLocaleString=True),
      smooth_factor=2.0
           
    ).add_to(m)



# m.add_child(folium.ClickForMarker(popup='point added'))
# #m.add_child(folium.LatLngPopup())

# folium.LatLngPopup().add_to(m)
# folium.LayerControl(collapsed = False).add_to(m)




plugins.Fullscreen(
    position="topright",
    title="Expand me",
    title_cancel="Exit me",
    force_separate_button=True,
).add_to(m)

minimap = plugins.MiniMap()
m.add_child(minimap)



# Add drawinf controls
draw = Draw()
draw.add_to(m)
m.add_child(MeasureControl())


folium.LayerControl().add_to(m)

# plugins.Geocoder().add_to(m)

m.save('D:\\LiLa_Nagapattinam\\Supporting_info\\Landsuitability.html')

# %%

top15 = read_df_UT("workdir\\top15.shp")

top15.reset_index(inplace =True)
top15
top15.level_0 = top15.level_0 + 1
top15= top15.round({"area_acres": 0})
top15.to_file('top15.json', driver='GeoJSON')
top15 = os.path.join('', 'top15.json')



m= folium.Map(location=[11.18, 79.7071], zoom_start=10)

loc = 'Top 15 lands'
title_html = '''
             <h3 align="center" style="font-size:16px"><b>{}</b></h3>
             '''.format(loc)   


m.get_root().html.add_child(folium.Element(title_html))


tile = folium.TileLayer(
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr = 'Esri',
        name = 'Esri Satellite',
        overlay = False,
        control = True
       ).add_to(m)

folium.GeoJson(
    top15,
     name='Top-15',
    show = False,
        style_function=lambda feature: {
        'fillColor':"#FBAF31",
        'color' : "#FBAF31",
        'weight' : 3,
        'fillOpacity' : 0.5,
        },
         highlight_function=lambda x: {'weight':5,'color':'yellow'},
         tooltip=folium.features.GeoJsonTooltip(fields=['level_0','area_acres'],aliases =["Rank :","Area (acres) :"],labels=True, toLocaleString=True)
      #smooth_factor=2.0
           
    ).add_to(m)

folium.GeoJson(
    _shp_district,
     name='dist_boundary',
    show = False,
    control =False,
        style_function=lambda feature: {
        'fillColor': "none",
        'color' : "black",
        'weight' : 1,
        'fillOpacity' : 0.5,
        },
#          highlight_function=lambda x: {'weight':5,'color':'yellow'},
#          tooltip=folium.features.GeoJsonTooltip(fields=['AREA'],labels=False, toLocaleString=True)
      #smooth_factor=2.0
           
    ).add_to(m)



plugins.Fullscreen(
    position="topright",
    title="Expand me",
    title_cancel="Exit me",
    force_separate_button=True,
).add_to(m)

minimap = plugins.MiniMap()
m.add_child(minimap)



# Add drawinf controls
draw = Draw()
draw.add_to(m)
m.add_child(MeasureControl())


folium.LayerControl().add_to(m)

# plugins.Geocoder().add_to(m)
m.save('D:\\LiLa_Nagapattinam\\Supporting_info\\top15lands.html')

# %%

# %%

# %%

# %%
