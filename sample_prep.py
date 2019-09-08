import rasterio as rio
from rasterio.plot import show
from rasterio.mask import mask
from shapely.geometry import mapping
import numpy as np
import os
import matplotlib as mpl
from matplotlib import pyplot as plt

import geopandas as gpd

plt.ion()

img_path = './landsat_data/LC08_214111_20181014.tif'
crroi_path = './sample/cr_shape.shp'
crimg_path = './sample/cr/'



#open landsat data
'''
with rio.open(img_path) as src:
    img = src.read(masked = True)
    extent = rio.plot.plotting_extent(src)
    img_profile = src.profile

fig, ax = plt.subplots(figsize = (10,10))
show(img,
     cmap='binary',
     ax=ax,
     extent=extent)
ax.set_title('landsat data', fontsize=16);
'''
#open roi shape
cr_roi = gpd.read_file(crroi_path)
with rio.open(img_path) as src:
    for i in range(0,len(cr_roi)):
    #for i in range(0,2):
        roi = mapping(cr_roi['geometry'][i])
        img_cr, cr_affine = mask(src,
                                 shapes=[roi],
                                 crop=True)
        img_cr_meta = src.meta.copy()
        img_cr_meta.update(height=int(img_cr.shape[1]), width=int(img_cr.shape[2]), nodata=0, transform=cr_affine, compress='lzw')
        with rio.open(crimg_path + str(i) + '.tif', 'w', **img_cr_meta) as ff:
            ff.write(img_cr[0], 1)


