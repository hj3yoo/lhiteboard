import cv2
import copy
from matplotlib import pyplot as plt
import math
import numpy as np
from scipy.spatial import distance

# Tuple of RGB values for special colours
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)


def find_coordinate(img, blur_size=5, threshold=50, percentile=10, pyramid_height=7, centre_ratio=0.25):
    """
    Core CV algorithm to locate the intended coordinate of the point given the reflections on the surface
    :param img: source image to run the algorithm on
    :param blur_size: size of the Gaussian blur mask
    :param threshold: threshold value of pixel brightness to be truncated
    :param percentile:
    :param pyramid_height: Number of levels for image pyramid
    :param centre_ratio: Size of the centre to be considered inside the detected ellipsoid
    :return: x, y coordinate of the point
    """
    # Convert image to grayscale, apply blurring and thresholding
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (blur_size, blur_size), 0)
    img_thresh = cv2.threshold(img_blur, threshold, 255, cv2.THRESH_TOZERO)[1]
    _, contours, _ = cv2.findContours(img_thresh, 1, 2)
    ellipses = [cv2.fitEllipse(cont) for cont in contours if len(cont) >= 5]
    if len(ellipses) == 0:
        #print('No ellipse found')
        return -1, -1
    # Sorted by the size of the ellipses in descending order
    def get_key(item):
        return item[1][0] * item[1][1]
    ellipses = sorted(ellipses, key=get_key, reverse=True)

    # Using the centre of the largest ellipse and brightest pixel's coordinate,
    # Compute new point's coordinate
    ellipse = ellipses[0]
    _, max_val, _, max_loc = cv2.minMaxLoc(img_thresh)
    #print(ellipse)
    #print(max_val, max_loc)
    new_pt_x = int((ellipse[0][0] + max_loc[0]) / 2 + 0.5)
    new_pt_y = int((ellipse[0][1] + max_loc[1]) / 2 + 0.5)
    new_pt = (new_pt_x, new_pt_y)

    # Mark the locations in the debug image
    #ellipse_centre = (int(ellipse[0][0] + 0.5), int(ellipse[0][1] + 0.5))
    #img_debug_show = cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2BGR)
    #img_debug_show = cv2.circle(img_debug_show, max_loc, 1, COLOR_RED, 1)
    #img_debug_show = cv2.ellipse(img_debug_show, ellipse, COLOR_GREEN, 1)
    #img_debug_show = cv2.circle(img_debug_show, ellipse_centre, 1, COLOR_RED, 1)
    #img_debug_show = cv2.circle(img_debug_show, new_pt, 1, COLOR_BLUE, 1)
    #cv2.imshow('coordinates', img_debug_show)
    #cv2.waitKey(0)
    return new_pt


def find_source(img, blur_size=5, threshold=40, neighbour_ratio=0.5):
    """
    Extract the important sections of images where light sources are found
    :param img: source image
    :param blur_size: size of Gaussian blur mask
    :param threshold: threshold value of pixel brightness to be truncated
    :param neighbour_ratio: percentage of spaces near the detected area to be considered
    :return: x, y, width, height of the source
    """
    ret = []
    # Filtering - convert to grayscale, apply Gaussian blur, then remove noises with threshold
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (blur_size, blur_size), 0)
    img_thresh = cv2.threshold(img_blur, threshold, 255, cv2.THRESH_TOZERO)[1]

    # Find all sources of IR light
    _, contours, _ = cv2.findContours(img_thresh, 1, 2)
    boxes = [cv2.boundingRect(cont) for cont in contours]
    for x, y, w, h in boxes:
        x0 = int(x - w * neighbour_ratio / 2)
        x1 = int(x + w * (1 + neighbour_ratio / 2))
        y0 = int(y - h * neighbour_ratio / 2)
        y1 = int(y + h * (1 + neighbour_ratio / 2))
        ret.append(((x0, y0), (x1, y1)))
        # cv2.imshow('source', img[y0:y1, x0:x1])
        # cv2.waitKey(0)

    #img_debug_show = cv2.resize(img_thresh, (len(img_thresh[0]) // 4, len(img_thresh) // 4))
    #cv2.imshow('threshold', img_debug_show)
    '''
    # NOTE: this snippet will omit all sources but the largest one
    # TODO: Convert to multi-coordinate detection
    # Sorted by the size of the rectangle
    def get_key(pt):
        return (pt[0][1] - pt[1][1]) * (pt[0][0] - pt[1][0])
    ret = sorted(ret, key=get_key, reverse=True)
    return [ret[0]]
    '''
    return ret
    pass
