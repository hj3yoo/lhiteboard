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


class Coord:
    # Simple struct to store 2D coordinate
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return '(%d, %d)' % (self.x, self.y)


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
    img = cv2.resize(img, (len(img[0]) * 4, len(img) * 4))

    # Convert image to grayscale, apply blurring and thresholding
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (blur_size, blur_size), 0)
    img_thresh = cv2.threshold(img_blur, threshold, 255, cv2.THRESH_TOZERO)[1]
    _, contours, _ = cv2.findContours(img_thresh, 1, 2)

    img_result = copy.copy(img)
    img_process = cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2RGB)
    img_ellipse = copy.copy(img_process)

    scale = percentile ** (1 / pyramid_height)
    brightness_pyramid = [calc_hist_percentile(img_thresh, threshold, 100 - pow(scale, x)) for x in range(1, pyramid_height + 1)]
    img_perc_pyramid = [cv2.threshold(img_thresh, top_n, 255, cv2.THRESH_TOZERO)[1] for top_n in brightness_pyramid]
    img_perc_pyramid_mask = [cv2.threshold(img_thresh, top_n, 255, cv2.THRESH_BINARY)[1] // 255 for top_n in brightness_pyramid]
    centroid = [find_centroid(img_perc_pyramid_mask[i]) for i in range(len(img_perc_pyramid)) if np.sum(img_perc_pyramid_mask[i]) != 0]
    #print(centroid)
    img_cont = np.sum([np.minimum((np.maximum(img_perc_pyramid[i], brightness_pyramid[i]) - brightness_pyramid[i]),
                                  256 // pyramid_height) for i in range(pyramid_height)], axis=0).astype(np.uint8)
    img_cont += img_thresh // (pyramid_height + 1)
    img_cont = cv2.cvtColor(img_cont, cv2.COLOR_GRAY2BGR)

    # The reflection of a light source over a flat surface can be estimated by an ellipsoidal shape
    # Estimate ellipses using the detected contours from thresholded image
    ellipses = [cv2.fitEllipse(cont) for cont in contours if len(cont) >= 5]

    # Sorted by the size of the ellipses in descending order
    def get_key(item):
        return item[1][0] * item[1][1]
    ellipses = sorted(ellipses, key=get_key, reverse=True)
    if len(ellipses) == 0:
        print('No ellipse found')
        return -1, -1

    '''
    split the largest ellipse into three sections:
    centre: a small ellipse located at the centre
    left: left half of the ellipse
    right: right half of the ellipse
    '''
    ellipse = ellipses[0]
    small_ellipse = (ellipse[0], (ellipse[1][0] * centre_ratio, ellipse[1][1] * centre_ratio), ellipse[2])

    #img_ellipse = cv2.ellipse(img_ellipse, ellipse, COLOR_GREEN, 1)

    x = ellipse[0][0]
    y = ellipse[0][1]
    width = ellipse[1][0]
    length = ellipse[1][1]
    theta = math.radians(ellipse[2])
    width_theta = (width / 2 * math.cos(theta), width / 2 * math.sin(theta))
    length_theta = (-1 * length / 2 * math.sin(theta), length / 2 * math.cos(theta))

    '''
    a------e------b
    |  ai  |  bi  |
    h----centre---f
    |  di  |  ci  |
    d------g------c
    '''
    a = (int(x + length_theta[0] - width_theta[0]), int(y + length_theta[1] - width_theta[1]))
    b = (int(x - length_theta[0] - width_theta[0]), int(y - length_theta[1] - width_theta[1]))
    c = (int(x - length_theta[0] + width_theta[0]), int(y - length_theta[1] + width_theta[1]))
    d = (int(x + length_theta[0] + width_theta[0]), int(y + length_theta[1] + width_theta[1]))
    e = (int(x - width_theta[0]), int(y - width_theta[1]))
    #f = (int(x - length_theta[0]), int(y - length_theta[1]))
    g = (int(x + width_theta[0]), int(y + width_theta[1]))
    #h = (int(x + length_theta[0]), int(y + length_theta[1]))

    #cv2.fillConvexPoly(img_process, np.array([a, e, g, d]), COLOR_BLUE)
    #cv2.fillConvexPoly(img_process, np.array([e, b, c, g]), COLOR_GREEN)
    #cv2.fillConvexPoly(img_process, np.array([ai, bi, ci, di]), COLOR_WHITE)
    #img_process = cv2.line(img_process, e, g, COLOR_WHITE)
    #img_process = cv2.line(img_process, f, h, COLOR_WHITE)
    #img_process = cv2.line(img_process, a, b, COLOR_WHITE)
    #img_process = cv2.line(img_process, b, c, COLOR_WHITE)
    #img_process = cv2.line(img_process, c, d, COLOR_WHITE)
    #img_process = cv2.line(img_process, d, a, COLOR_WHITE)
    #img_process = cv2.ellipse(img_process, small_ellipse, COLOR_WHITE, -1)

    for i in range(len(centroid)):
        img_process = cv2.circle(img_process, centroid[i], 3, (0, 0, 255), 1)

    img_blank = cv2.threshold(img, 255, 255, cv2.THRESH_BINARY)[1]
    img_thresh_rgb = cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2BGR)
    img_small_ellipse = cv2.ellipse(copy.copy(img_blank), small_ellipse, COLOR_WHITE, -1)
    img_small_ellipse = cv2.bitwise_and(img_thresh_rgb, img_small_ellipse)
    _, contours_small, _ = cv2.findContours(cv2.cvtColor(img_small_ellipse, cv2.COLOR_BGR2GRAY), 1, 2)
    inside_centre = [cv2.pointPolygonTest(contours_small[0], centroid[i], False) for i in range(len(centroid))]
    #print(inside_centre)
    img_left_ellipse = copy.copy(img_blank)
    img_right_ellipse = copy.copy(img_blank)
    if len(inside_centre) > 0 and inside_centre[0] == 1:
        print('Inside centre')
        pt = centroid[0]
    else:
        print('Outside circle')
        img_left_ellipse = cv2.fillConvexPoly(img_left_ellipse, np.array([a, e, g, d]), COLOR_WHITE)
        img_right_ellipse = cv2.fillConvexPoly(img_right_ellipse, np.array([e, b, c, g]), COLOR_WHITE)
        img_left_ellipse = cv2.bitwise_and(img_thresh_rgb, img_left_ellipse)
        img_right_ellipse = cv2.bitwise_and(img_thresh_rgb, img_right_ellipse)
        left_sum = np.sum(img_left_ellipse)
        right_sum = np.sum(img_right_ellipse)
        print(left_sum, right_sum)

        if left_sum > right_sum:
            thresh_bright = calc_hist_percentile(img_left_ellipse, threshold, 100 - percentile)
            img_percentile_mask = cv2.threshold(img_left_ellipse, thresh_bright, 255, cv2.THRESH_BINARY)[1]
            pt = find_centroid(cv2.cvtColor(img_percentile_mask, cv2.COLOR_BGR2GRAY))
            #print(thresh_bright, pt)
        elif left_sum < right_sum:
            thresh_bright = calc_hist_percentile(img_right_ellipse, threshold, 100 - percentile)
            img_percentile_mask = cv2.threshold(img_right_ellipse, thresh_bright, 255, cv2.THRESH_BINARY)[1]
            pt = find_centroid(cv2.cvtColor(img_percentile_mask, cv2.COLOR_BGR2GRAY))
            #print(thresh_bright, pt)
        else:
            pt = centroid[0]
    img_result = cv2.circle(img_result, pt, len(img) // 50, COLOR_GREEN, len(img) // 100 + 1)
    ret = pt
    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = len(img) / 200
    text_pt = (0, len(img_result[0]) - len(img[0]) // 10)
    cv2.putText(img_result, 'Result', text_pt, font, font_scale, COLOR_WHITE)
    cv2.putText(img_ellipse, 'Detection Area', text_pt, font, font_scale, COLOR_WHITE)
    cv2.putText(img_cont, 'Contrast Enhanced', text_pt, font, font_scale, COLOR_WHITE)
    cv2.putText(img_left_ellipse, 'Left Ellipse', text_pt, font, font_scale, COLOR_WHITE)
    cv2.putText(img_right_ellipse, 'Right Ellipse', text_pt, font, font_scale, COLOR_WHITE)
    cv2.putText(img_process, 'Processing Boundary', text_pt, font, font_scale, COLOR_WHITE)

    img_out = np.vstack((np.hstack((img_result, img_ellipse, img_cont)),
                         np.hstack((img_left_ellipse, img_right_ellipse, img_process))))

    print(ret)
    #img_out = cv2.resize(img_out, (min(len(img_out[0]), 800), min(len(img_out), int(800 * len(img_out) / len(img_out[0])))))
    cv2.imshow('Output', img_out)
    cv2.waitKey(0)
    return ret


def find_centroid(img):
    """
    Find the centroid of the blob using brightness as a weight
    :param img: source image with the blob
    :return: x, y coordinate of the centroid
    """
    count_pixel = np.sum(img)
    if count_pixel == 0:
        raise ValueError('no pixels exists to find centroid')
    mat_row_ind = np.array([[i for i in range(len(img[0]))] for _ in range(len(img))])
    mat_col_ind = np.array([[i for _ in range(len(img[0]))] for i in range(len(img))])
    avg_row_ind = int(np.sum(mat_row_ind * img) / count_pixel)
    avg_col_ind = int(np.sum(mat_col_ind * img) / count_pixel)
    return avg_row_ind, avg_col_ind


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
    num_counted = 0
    # Starting from the lowest brightness, count the number of pixels in each bin until the given percentile
    for idx in range(len(hist)):
        num_pixel = hist[idx][0]
        num_counted += num_pixel
        if num_counted >= thresh_num_pixel:
            return idx + min_bright - 1
    raise ValueError('Percentile not found')


def find_source(img, blur_size=5, threshold=40, neighbour_ratio=0.5):
    """
    Extract the important sections of images where light sources are found
    :param img: source image
    :param blur_size: size of Gaussian blur mask
    :param threshold: threshold value of pixel brightness to be truncated
    :param neighbour_ratio: percentage of spaces near the detected area to be considered
    :return: x, y, width, height of the
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
        #cv2.imshow('source', img[y0:y1, x0:x1])
        #cv2.waitKey(0)
    return ret


def rect_distance(rect_a, rect_b):
    """
    Computes the shortest distance between two rectangles
    Each rectangles are represented by list of list - [[x0,y0],[x1,y1]]
    :param rect_a: the primary rectangle
    :param rect_b: the secondary rectangle to be compared
    :return: distance in float
    """
    a1, a2 = rect_a
    b1, b2 = rect_b

    # Identify the position of rect_b with respect to rect_a
    left = b2[0] < a1[0]
    right = a2[0] < b1[0]
    bottom = b2[1] < a1[1]
    top = a2[1] < b1[1]

    # Calculate the distance considering the positional relationship betwen two rectangles
    if top and left:
        return distance.euclidean((a1[0], a2[1]), (b2[0], b1[1]))
    elif left and bottom:
        return distance.euclidean((a1[0], a1[1]), (b2[0], b2[1]))
    elif bottom and right:
        return distance.euclidean((a2[0], a1[1]), (b1[0], b2[1]))
    elif right and top:
        return distance.euclidean((a2[0], a2[1]), (b1[0], b1[1]))
    elif left:
        return a1[0] - b2[0]
    elif right:
        return b1[0] - a2[0]
    elif bottom:
        return a1[1] - b2[1]
    elif top:
        return b1[1] - a2[1]
    else:             # rectangles intersect
        return 0.
