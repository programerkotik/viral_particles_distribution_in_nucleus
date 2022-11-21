from tifffile import imread
import cv2 as cv
import numpy as np

def read_data(path, viewer):
    ''' Input: path to the data file
        Output: 4 separate channels of image data
        Function also send the data to napari viewer '''
    data = imread(path)
    # convert to 8-bit
    data = (data / data.max() * 255).astype('uint8')
    viewer.add_image(data[0], name='STAR RED_CONF', colormap='red', blending='additive', contrast_limits=[0, 150])
    viewer.add_image(data[1], name='STAR 580_CONF', colormap='green', blending='additive', contrast_limits=[0, 150])
    viewer.add_image(data[2], name='STAR RED_STED', colormap='red', blending='additive', contrast_limits=[0, 150])
    viewer.add_image(data[3], name='STAR 580_STED', colormap='green', blending='additive', contrast_limits=[0, 150])
    return data[0], data[1], data[2], data[3]

def threshold(imgs,chs, viewer, kernel_size=5):
    ''' Input: confolcal red channel, kernel size
        Output: thresholded red channel '''
    ths = []
    for img, ch in zip(imgs, chs):
        # use medianBlur filter to smooth the image
        img = cv.medianBlur(img,kernel_size)
        # threshold the image
        __,th = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        viewer.add_labels(th, name=ch+'_th')
        ths.append(th)
    return ths

def generate_membrane(nucleus, viewer):
    ''' Input: thresholded nucleus
        Output: membrane  '''
    # get contour of the nucleus
    membrane = np.zeros(nucleus.shape)
    contours, __ = cv.findContours(nucleus, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # draw the contour on the membrane
    cv.drawContours(membrane, contours, -1, 255, 1)
    # convert to uint8
    membrane = membrane.astype('uint8')
    viewer.add_labels(membrane, name='Lamin skeleton')
    return membrane

def find_furthest_point(centroid, membrane_points):
    ''' Input: centroid of the nucleus, membrane points
        Output: furthest point from the centroid '''
    # find the furthest point from the centroid
    dist = np.sqrt((membrane_points[:,0] - centroid[0])**2 + (membrane_points[:,1] - centroid[1])**2)
    furthest_point = membrane_points[np.argmax(dist)]
    return furthest_point

def find_closest_point(point, membrane_points):
    ''' Input: point, membrane points
        Output: closest point to the membrane_point '''
    # find the closest point to the furthest point
    dist = np.sqrt((membrane_points[:,0] - point[0])**2 + (membrane_points[:,1] - point[1])**2)
    closest_point = membrane_points[np.argmin(dist)]
    return closest_point

def compute_distances(points, pixel_size_side, dist_for_normalization, closest_points):
    ''' Input: points, pixel size, distance for normalization, closest points of the membrane
        Output: distances between points and membrane '''
    # compute distances between points and membrane
    distances = []
    for particle, membrane_closest_point in zip(points, closest_points):
        dist = np.sqrt((membrane_closest_point[0] - particle[0])**2 + (membrane_closest_point[1] - particle[1])**2)
        distances.append(dist * pixel_size_side)
    normalized_distances = np.array(distances) / dist_for_normalization
    return normalized_distances, np.mean(normalized_distances), np.std(normalized_distances), distances, np.mean(distances), np.std(distances)

def generate_points(nucleus, n_points):
    ''' Input: nucleus, number of points
        Output: points '''
    # get all coords of the nucleus
    all_points = np.where(nucleus)
    # randomly sample n_points from the nucleus with no repeats
    points = np.array([np.random.choice(all_points[0], n_points, replace=False), np.random.choice(all_points[1], n_points, replace=False)]).T
    return points

def compute_pcc(ch1,ch2,nucleus):
    ''' Input: compute Pearson correlation coefficient between two channels 
        Output: Pearson correlation coefficient '''
    # get only point inside the nucleus
    ch1 = ch1[nucleus]
    ch2 = ch2[nucleus]
    # compute Pearson correlation coefficient
    numerator = np.sum((ch1 - np.mean(ch1)) * (ch2 - np.mean(ch2)))
    denominator = np.sqrt(np.sum((ch1 - np.mean(ch1))**2) * np.sum((ch2 - np.mean(ch2))**2))
    pcc = numerator / denominator
    return pcc
