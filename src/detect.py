import cv2
import copy
from matplotlib import pyplot as plt
import math
import numpy as np
from datetime import datetime

# Tuple of RGB values for special colours
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)


def find_coordinate(img, blur_size=5, static_threshold=30, dynamic_threshold=50, debug=False, show_image=False):
    """
    Core CV algorithm to locate the intended coordinate of the point given the reflections on the surface
    :param img: source image to run the algorithm on
    :param blur_size: size of the Gaussian blur mask
    :param static_threshold: static value of threshold to remove ambient noises
    :param dynamic_threshold: percentage of pixels ordered in brightness to be selected from thresholding
    :param debug:
    :param show_image:
    :return: x, y coordinate of the point, and elapsed time (for debugging)
    """
    if debug:
        t_start = datetime.now()
    else:
        t_elapsed = -1.0
    # If the cropped image is too small to do any meaningful process, just return the centre of it
    if img.shape[0] <= blur_size // 2 and img.shape[1] <= blur_size // 2:
        if debug:
            t_end = datetime.now()
            t_elapsed = (t_end - t_start).total_seconds()
        return (img.shape[0] // 2, img.shape[1] // 2), t_elapsed
    # Pre-processing - convert to grayscale, apply Gaussian blur, then remove noises with threshold
    # Check if the image is already grayscale
    if len(img.shape) == 2:
        img_gray = img
    else:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (blur_size, blur_size), 0)
    # With pixels brighter than static threshold value x (to cancel out noises),
    # Find the top n% bright pixels
    thresh_value = calc_hist_percentile(img, static_threshold, 100 - dynamic_threshold)
    img_thresh = cv2.threshold(img_blur, thresh_value, 255, cv2.THRESH_TOZERO)[1]

    # Run contour detection, then each of them into an elliptical shape
    _, contours, _ = cv2.findContours(img_thresh, 1, 2)
    ellipses = [cv2.fitEllipse(cont) for cont in contours if len(cont) >= 5]
    if len(ellipses) == 0:
        if debug:
            t_end = datetime.now()
            t_elapsed = (t_end - t_start).total_seconds()
        return (-1, -1), t_elapsed

    # Sorted by the size of the ellipses in descending order
    def get_key(item):
        return item[1][0] * item[1][1]
    ellipses = sorted(ellipses, key=get_key, reverse=True)

    # Using the centre of the largest ellipse and brightest pixel's coordinate,
    # Compute new point's coordinate
    ellipse = ellipses[0]
    _, max_val, _, max_loc = cv2.minMaxLoc(img_thresh)
    # Weighted average is used between two coordinates
    ellipse_weight = 0.5
    brightest_weight = 1 - ellipse_weight
    new_pt_x = int(ellipse[0][0] * ellipse_weight + max_loc[0] * brightest_weight + 0.5)
    new_pt_y = int(ellipse[0][1] * ellipse_weight + max_loc[1] * brightest_weight + 0.5)
    new_pt = (new_pt_x, new_pt_y)

    if debug:
        t_end = datetime.now()
        t_elapsed = (t_end - t_start).total_seconds()
    if show_image:
        # Mark the locations in the debug image
        ellipse_centre = (int(ellipse[0][0] + 0.5), int(ellipse[0][1] + 0.5))
        img_debug_show = cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2BGR)
        img_debug_show = cv2.circle(img_debug_show, max_loc, 1, COLOR_RED, 1)
        img_debug_show = cv2.ellipse(img_debug_show, ellipse, COLOR_GREEN, 1)
        img_debug_show = cv2.circle(img_debug_show, ellipse_centre, 1, COLOR_RED, 1)
        img_debug_show = cv2.circle(img_debug_show, new_pt, 1, COLOR_BLUE, 1)
        cv2.imshow('coordinates', img_debug_show)
        #cv2.waitKey(0)
    return new_pt, t_elapsed


def find_source(img, blur_size=5, threshold=40, neighbour_ratio=0.5, debug=False, show_image=False):
    """
    Extract the important sections of images where light sources are found
    :param img: source image
    :param blur_size: size of Gaussian blur mask
    :param threshold: threshold value of pixel brightness to be truncated
    :param neighbour_ratio: percentage of spaces near the detected area to be considered
    :param debug:
    :param show_image:
    :return: x, y, width, height of the source
    """
    if debug:
        t_start = datetime.now()
    else:
        t_elapsed = -1.0
    ret = []
    if img is None:
        if debug:
            t_end = datetime.now()
            t_elapsed = (t_end - t_start).total_seconds()
        return ret, t_elapsed
    # Pre-processing - convert to grayscale, apply Gaussian blur, then remove noises with threshold
    # Check if the image is already grayscale
    if len(img.shape) == 2:
        img_gray = img
    else:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (blur_size, blur_size), 0)
    img_thresh = cv2.threshold(img_blur, threshold, 255, cv2.THRESH_TOZERO)[1]

    # Find all sources of IR light
    # Run contour detection, then each of them into a rectangular box
    _, contours, _ = cv2.findContours(img_thresh, 1, 2)
    boxes = [cv2.boundingRect(cont) for cont in contours]
    for x, y, w, h in boxes:
        x0 = max(0, int(x - w * neighbour_ratio / 2))
        x1 = min(img.shape[0], int(x + w * (1 + neighbour_ratio / 2)))
        y0 = max(0, int(y - h * neighbour_ratio / 2))
        y1 = min(img.shape[1], int(y + h * (1 + neighbour_ratio / 2)))
        ret.append(((x0, y0), (x1, y1)))
        '''
        #if debug:
        #    cv2.imshow('source', img[y0:y1, x0:x1])
        #    cv2.waitKey(0)
        '''

    if debug:
        t_end = datetime.now()
        t_elapsed = (t_end - t_start).total_seconds()
    if show_image:
        img_debug_show = cv2.drawContours(img_thresh, contours, -1, COLOR_BLUE, 3)
        img_debug_show = cv2.resize(img_thresh, (len(img_thresh[0]) // 4, len(img_thresh) // 4))
        cv2.imshow('threshold', img_debug_show)
        #cv2.waitKey(0)

    # Sorted by the size of the rectangle in descending order
    def get_key(pt):
        return (pt[0][1] - pt[1][1]) * (pt[0][0] - pt[1][0])
    ret = sorted(ret, key=get_key, reverse=True)

    if debug:
        t_end = datetime.now()
        t_elapsed = (t_end - t_start).total_seconds()
    return ret, t_elapsed
    pass

def calc_hist_percentile(img, min_bright, percentile):
    """
    Find the brightness value of pixels for the given percentile
    :param img: source image
    :param min_bright: any pixels below this brightness value will be ignored
    :param percentile: percentile value to decide the threshold
    :return: threshold brightness value
    """
    # Create a histogram where each possible brightness value is represented by a bin
    hist = cv2.calcHist([img], [0], None, [256 - min_bright], [min_bright, 256])
    sample_size = sum(hist)
    thresh_num_pixel = sample_size * percentile / 100
    #print(sample_size, thresh_num_pixel)
    num_counted = 0
    # Starting from the lowest brightness, count the number of pixels in each bin until the given percentile
    for idx in range(len(hist)):
        num_pixel = hist[idx][0]
        num_counted += num_pixel
        if num_counted >= thresh_num_pixel:
            return idx + min_bright - 1
    raise ValueError('Percentile not found')
