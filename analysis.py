#%%
import numpy as np
from tifffile import imread
import napari
import matplotlib.pyplot as plt
import cv2 as cv
from skimage import measure
from utils import *
from scipy import ndimage
from skimage.morphology import skeletonize
from math import sqrt
from skimage.feature import blob_log
from skimage import morphology

viewer = napari.Viewer() # run napari

#%%
# import data
name = '05_mxlaminAC-635P_rxVP1-580'
ch1,ch2,ch3,ch4 = read_data(f'data_to_use/2022-02-17_deconvolved_tiff/{name}.ome.tif', viewer)

# %%
# visualize segmented lamin
membrane = ch2
kernel = (15,15)
threshold(membrane, kernel, viewer)
# create a mask for nucleus
segment_nucleus(viewer)

# %%
# manually edit nucleus and save it as variable
nucleus = viewer.layers['nucleus'].data

# %%
# create skeleton of membrane
membrane = viewer.layers['STAR RED_CONF_label'].data
skeleton = skeletonize(membrane/membrane.max())
skeleton = skeleton.astype(np.uint8)
viewer.add_labels(skeleton * randint(0,255), name = 'Lamin skeleton')

#%% 
skeleton = viewer.layers['Lamin skeleton'].data

#%%
# calculate value for normalization 
nucleus = viewer.layers['nucleus'].data
x,y = np.where(nucleus==1)
centroid = (np.mean(x), np.mean(y))

viewer.add_points(centroid)

# find shorted path between centroid and skeleton
membrane_points = np.array(np.where(skeleton)).T
furthest_point = find_furthest_point(centroid, membrane_points)

dist_for_normalization = np.linalg.norm(np.subtract(centroid,furthest_point))
viewer.add_shapes([furthest_point, centroid], shape_type='line', edge_color='red')

#%%
# show protein
protein = viewer.layers['STAR 580_STED'].data
lamin = viewer.layers['STAR RED_STED'].data
only_protein = np.subtract(protein.astype(float),lamin.astype(float))
only_protein = np.clip(only_protein, 0, 255)
particles = np.where(nucleus != nucleus.max(), 0, only_protein)
particles = np.array(particles, dtype = np.uint8)
viewer.add_image(particles)

# %%
# blob detection
blobs_log = blob_log(particles, max_sigma=5, num_sigma=10, threshold=0.15)
blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
points = blobs_log[:,:2]
viewer.add_points(points, size=5, symbol='ring', face_color='red')

#%%
# analyse distances
membrane_points = np.array(np.where(skeleton)).T

closest_points = []
for point in points: 
    closest_point = find_closest_point(point, membrane_points)
    closest_points.append(closest_point)

closest_points = np.array(closest_points)

# %%
# show distances
distances = np.empty((points.shape[0], 2, 2))
distances[:, 0] = points
distances[:, 1] = closest_points

viewer.add_shapes(distances, shape_type='line', edge_color='red') 
# %%
# compute distances
pixel_size_side = 0.02 # micrometers
distances = []
for point, closest_point in zip(points,closest_points):
    point = pixel_size_side * point
    closest_point = pixel_size_side * closest_point
    dist = np.linalg.norm(np.subtract(point,closest_point))
    distances.append(dist/ dist_for_normalization)

mean = np.mean(distances)
std = np.std(distances)

plt.hist(distances, bins = 100)
plt.ylabel('Number of protein particles')
plt.xlabel('Distance (in $\mu m$)')
plt.title(f'Distribution of distances: $\mu={round(mean, 4)},\ \sigma={round(std, 4)}$')
plt.savefig(f'{name}.png')

# %%
# write results into some file 
from pathlib import Path
import csv
from os.path import exists

distances = np.array(distances)
np.savetxt(f"{name}.csv", distances, delimiter=",")

# %%
