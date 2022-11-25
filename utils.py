from tifffile import imread
import cv2 as cv
import numpy as np
import os

def read_data(path, viewer):
    ''' Input: path to the data file
        Output: 4 separate channels of image data
        Function sends the data to napari viewer '''
    data = imread(path)
    data = (data / data.max() * 255).astype('uint8') # convert to 8-bit
    image_data = [data[0], data[1], data[2], data[3]]
    channels = ['STAR RED_CONF', 'STAR 580_CONF', 'STAR RED_STED', 'STAR 580_STED']
    colors = ['red', 'green', 'red', 'green']
    for img, ch, color in zip(image_data, channels, colors):
        viewer.add_image(img, name=ch, colormap=color, blending='additive', contrast_limits=[0, 150])
    return data[0], data[1], data[2], data[3]

def threshold(imgs,chs, viewer, kernel_size=5):
    ''' Input: confolcal red channel, kernel size
        Output: thresholded red channel '''
    ths = []
    for img, ch in zip(imgs, chs):
        img = cv.medianBlur(img,kernel_size) # use medianBlur filter to smooth the image
        __,th = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU) # threshold the image
        ths.append(th)
        viewer.add_labels(th, name=ch+'_th')
    return ths

def generate_membrane(nucleus, viewer):
    ''' Input: thresholded nucleus
        Output: membrane  '''
    membrane = np.zeros(nucleus.shape)
    contours, __ = cv.findContours(nucleus, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) # get contour of the nucleus
    cv.drawContours(membrane, contours, -1, 255, 1) # draw the contour on the membrane
    membrane = membrane.astype('uint8') # convert to uint8
    viewer.add_labels(membrane, name='Lamin skeleton')
    return membrane

def find_furthest_point(ctr, xs):
    ''' Input: centroid of the nucleus, membrane points
        Output: furthest point from the centroid '''
    dist = np.sqrt((xs[:,0] - ctr[0])**2 + (xs[:,1] - ctr[1])**2)
    return xs[np.argmax(dist)]

def find_closest_point(ctr, xs):
    ''' Input: point, membrane points
        Output: closest point to the membrane_point '''
    dist = np.sqrt((xs[:,0] - ctr[0])**2 + (xs[:,1] - ctr[1])**2)
    return xs[np.argmin(dist)]

def compute_distances(xs, ys, pixel_size, norm_dist):
    ''' Input: points, pixel size, distance for normalization, closest points of the membrane
        Output: distances between points and membrane '''
    dists = [np.sqrt((y[0] - x[0])**2 + (y[1] - x[1])**2) for x,y in zip(xs*pixel_size,ys*pixel_size)] # compute distances between points and membrane
    norm_dists = np.array(dists) / norm_dist
    return norm_dists, np.mean(norm_dists), np.std(norm_dists), dists, np.mean(dists), np.std(dists)

def generate_points(nucleus, n):
    ''' Input: nucleus, number of points
        Output: points '''
    all_points = np.array(np.where(nucleus)).T # get all coords of the nucleus
    ids = np.random.choice(range(len(all_points)), n, replace=False) # randomly sample n_points from the nucleus with no repeats
    points = all_points[ids]
    return points

def compute_pcc(ch1,ch2,nucleus):
    ''' Input: compute Pearson correlation coefficient between two channels 
        Output: Pearson correlation coefficient '''
    # get only points inside the nucleus
    ch1 = ch1[nucleus]
    ch2 = ch2[nucleus]
    # compute Pearson correlation coefficient
    numerator = np.sum((ch1 - np.mean(ch1)) * (ch2 - np.mean(ch2)))
    denominator = np.sqrt(np.sum((ch1 - np.mean(ch1))**2) * np.sum((ch2 - np.mean(ch2))**2))
    pcc = numerator / denominator
    return pcc

def find_csv_directories(directory, name):
    ''' Input: directory, name of the csv file
        Output: list of csv directories and list of group names '''
    groups = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))] # find all group folders
    # find all csv files in each group folder
    csv_directories = []
    for group in groups:
        group_csv = []
        root_dir = os.path.join(directory, group)
        # walk through all folders in the group folder
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.csv'):
                    group_csv.append(os.path.join(root, file))
        csv_directories.append(group_csv)
    return csv_directories, groups
