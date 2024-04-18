from rasterio import features
from shapely.geometry import shape
import rasterio
import pickle
import numpy as np
from rasterio import mask
import geopandas as gpd
import fiona
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from shapely import MultiPolygon, Polygon
from shapely.geometry import box
import json
from PIL import Image
import rioxarray

f = open('ignitions.json')
ignitions = json.load(f)
dir_list = os.listdir("./hour_graphs/") 
isoc = {}
max_box = [0,0]
centers = {}
bigs = {}
for i in tqdm(dir_list):
    fire_n = i.split("_")[1].split('.')[0]
    with open(f"../data/hour_graphs/{i}", "rb") as file:
        hr_graph = pickle.load(file)
        if len(hr_graph) > 1:
            with rasterio.open('./landscape/Input_Geotiff.tif') as f:
                image = f.read(1)
                transform = f.transform
                crs = f.crs
                # use f.nodata if possible; it's not defined on this particular image
                nodata = -9999.0
                dims = f.read(1).shape
                mask = np.zeros(dims, dtype=np.bool_).astype(np.uint8)
                idx = np.unravel_index(ignitions[fire_n] - 1, (1173, 1406))
                assert(image[(idx[0]),(idx[1])] != nodata)
                mask[(idx[0]),(idx[1])] = True
                for t in hr_graph:
                    for k,j in hr_graph[t][0]:
                        idx = np.unravel_index(k - 1, (1173, 1406))
                        mask[(idx[0]),(idx[1])] = True
                        assert(image[(idx[0]),(idx[1])] != nodata)
                        idx2 = np.unravel_index(j - 1, (1173, 1406))
                        mask[(idx2[0]),(idx2[1])] = True
                        assert(image[(idx[0]),(idx[1])] != nodata)
                polygons = []
                for coords, value in features.shapes(mask, transform = transform):
                    # ignore polygons corresponding to nodata
                    if value != 0:
                        # convert geojson to shapely geometry
                        geom = shape(coords)
                        polygons.append(geom)
                # use the feature loop in case you polygon is a multipolygon
                if mask.sum():
                    features_ = [0]
                    multi_p = MultiPolygon(polygons) # add crs using wkt or EPSG to have a .prj file
                    min_x, min_y, max_x, max_y = multi_p.bounds
                    bounding_box = ((max_x - min_x) + 320, (max_y - min_y) + 320)
                    if bounding_box[0] > max_box[0]:
                        max_box[0] = bounding_box[0]
                        if max_box[0]//80 > 500:
                            print("Found a big one (x)!")
                            bigs[i] = max_box
                    if bounding_box[1] > max_box[1]:
                        max_box[1] = bounding_box[1]
                        if max_box[1]//80 > 500:
                            print("Found a big one (y)!")
                            bigs[i] = max_box
                    centers[fire_n] = (min_x + (max_x - min_x)/2, min_y + (max_y - min_y)/2)

bbox = max_box
pad = 80
with rasterio.open('./landscape/Input_Geotiff.tif') as f:
    limits = f.bounds
    limits_x = (limits[0], limits[2])
    limits_y = (limits[1], limits[3])
    dims = f.read(1).shape
    image = f.read(1)
    transform = f.transform



for i in tqdm(centers):
    x, y = centers[i][0], centers[i][1]
    max_x = x + (bbox[0]/2)
    min_x = x - (bbox[0]/2)
    max_y = y + (bbox[1]/2)
    min_y = y - (bbox[1]/2)
    if max_x > limits[2]:
        delta = limits[2] - max_x
        max_x += delta
        min_x += delta
    if max_y > limits[3]:
        delta = limits[3] - max_y
        max_y += delta
        min_y += delta
    if min_x < limits[0]:
        delta = limits[0] - min_x
        max_x += delta
        min_x += delta
    if min_y < limits[1]:
        delta = limits[1] - min_y
        max_y += delta
        min_y += delta
    indices = ((bbox[1]))
    ## CHECK:
    if max_x > limits_x[1] or max_y > limits_y[1] or min_x < limits_x[0] or min_y < limits_y[0]:
        print(max_x, min_x, max_y, min_y)
        raise Exception("Out of bounds!")  
    # box(minx, miny, maxx, maxy, ccw=True)
    geom = box(min_x, min_y, max_x, max_y)
    gdr = gpd.GeoDataFrame({'feature': features_, 'geometry': geom}, crs=crs)
    gdr.to_file(f"./shapes/box_{i}.shp")
    with open(f"../data/hour_graphs/graph_{i}.pkl", "rb") as file:
        hr_graph = pickle.load(file)
    with fiona.open(f"./shapes/box_{i}.shp", "r") as shapefile:
            shapes = [feature["geometry"] for feature in shapefile]
    with rioxarray.open_rasterio(f"./landscape/Input_Geotiff.tif") as src:
            out_image = src.rio.clip(shapes).values
            out_image = np.where(out_image == -9999.0, -1, out_image)
    if len(hr_graph) > 1:
        array_list = [out_image[i] for i in range(out_image.shape[0])]
        np.savez_compressed(f'backgrounds/background_{i}.npz', a1 = array_list[0]
                            , a2 = array_list[1], a3 = array_list[2], a4 = array_list[3]
                            , a5 = array_list[4], a6 = array_list[5], a7 = array_list[6]
                            , a8 = array_list[7])
        nodata = -9999.0
        mask = np.zeros(dims, dtype=np.bool_).astype(np.uint8)
        idx = np.unravel_index(ignitions[i] - 1, (1173, 1406))
        mask[(idx[0]),(idx[1])] = True
        shape = (int(max_box[1])//80, int(max_box[0]//80))
        mask_ = np.zeros(shape, dtype=np.bool_).astype(np.uint8)
        coords = rasterio.transform.xy(transform, idx[0], idx[1])
        x = int((coords[0] - min_x) / 80)
        y = int((max_y - coords[1]) / 80)
        # print(f"Marking ignition point for fire {ignitions[i]}, in pos ({y}, {x})")
        mask_[y,x] = True
        temp = 0
        for t in hr_graph:
            while t > temp:
                #plt.imsave(f'spreads/fire_{i}-{temp}.png', mask_)
                #plt.imsave(f'spreads/iso_{i}-{temp}.png', iso_)
                temp += 1
            iso = np.zeros(dims, dtype=np.bool_).astype(np.uint8)
            iso_ = np.zeros(shape)
            for k,j in hr_graph[t][0]:
                idx = np.unravel_index(k - 1, (1173, 1406))
                mask[(idx[0]),(idx[1])] = True
                iso[(idx[0]),(idx[1])] = True
                coords = rasterio.transform.xy(transform, idx[0], idx[1])
                x = int((coords[0] - min_x) / 80)
                y = int((max_y - coords[1]) / 80)
                mask_[y,x] = True
                iso_[y,x] = True
                idx2 = np.unravel_index(j - 1, (1173, 1406))
                mask[(idx2[0]),(idx2[1])] = True
                iso[(idx2[0]),(idx2[1])] = True
                coords = rasterio.transform.xy(transform, idx2[0], idx2[1])
                x = int((coords[0] - min_x) / 80)
                y = int((max_y - coords[1]) / 80)
                mask_[y,x] = True
                iso_[y,x] = True
            assert(mask_.sum() != 0)
            #plt.imsave(f'spreads/fire_{i}-{t}.png', mask_)
            #plt.imsave(f'spreads/iso_{i}-{t}.png', iso_)
            temp = t + 1
    