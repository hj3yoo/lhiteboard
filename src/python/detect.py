import os.path
import cv2

import copy
from matplotlib import pyplot as plt
import numpy as np
import sys

class Coord:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return '(%d, %d)' % (self.x, self.y)


def detect_blob(img):
    threshold = 50
    climit = 15
    clahe = cv2.createCLAHE(clipLimit=climit, tileGridSize=(10,10))
    edge_detection_kernel = np.array([[-1, -1, -1, -1, -1], [-1, 1, 1, 1, -1], [-1, 1, 8, 1, -1], [-1, 1, 1, 1, -1], [-1, -1, -1, -1, -1]])
    #edge_detection_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    blur_kernel = np.ones((5, 5)) / 25
    # Filter the image with Gaussian blur & binary threshold
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_thresh = cv2.threshold(img_blur, threshold, 255, cv2.THRESH_TOZERO)[1]

    pyramid_height = 15
    perc_thresh = 80
    scale = perc_thresh ** (1 / pyramid_height)
    print(scale)
    brightness_pyramid = [calc_hist_percentile(img_thresh, threshold, 100 - pow(scale, x)) for x in range(1, pyramid_height + 1)]
    img_perc_pyramid = [cv2.threshold(img_thresh, top_n, 255, cv2.THRESH_TOZERO)[1] for top_n in brightness_pyramid]
    img_perc_pyramid_mask = [cv2.threshold(img_thresh, top_n, 255, cv2.THRESH_BINARY)[1] // 255 for top_n in brightness_pyramid]
    count_pixel_pyramid = [np.sum(mask) for mask in img_perc_pyramid_mask]

    mat_row_ind = np.array([[i for i in range(len(img_thresh))] for _ in range(len(img_thresh[0]))])
    mat_col_ind = np.array([[i for _ in range(len(img_thresh))] for i in range(len(img_thresh[0]))])
    avg_row_ind = [int(np.sum(mat_row_ind * img_perc_pyramid_mask[i]) / count_pixel_pyramid[i]) for i in range(len(img_perc_pyramid)) if count_pixel_pyramid[i] != 0]
    avg_col_ind = [int(np.sum(mat_col_ind * img_perc_pyramid_mask[i]) / count_pixel_pyramid[i]) for i in range(len(img_perc_pyramid)) if count_pixel_pyramid[i] != 0]
    print(avg_row_ind)
    print(avg_col_ind)

    img_cont = np.sum([np.minimum((np.maximum(img_perc_pyramid[i], brightness_pyramid[i]) - brightness_pyramid[i]),
                                  256 // pyramid_height) for i in range(pyramid_height)], axis=0).astype(np.uint8)
    img_cont += img_thresh // (pyramid_height + 1)
    #img_cont = clahe.apply(img_cont)
    #img_cont = cv2.GaussianBlur(img_cont, (5, 5), 0)

    #img_thresh2 = cv2.threshold(img_blur, threshold, 255, cv2.THRESH_TRUNC)[1]
    #img_edge = cv2.filter2D(img_thresh2, -1, edge_detection_kernel)
    #img_blur2 = np.maximum(cv2.filter2D(img_edge, -1, blur_kernel), img_edge)
    #img_dilate = cv2.dilate(img_blur2, np.ones((5, 5)))
    #img_blur2 = cv2.filter2D(img_edge, -1, blur_kernel)
    #img_edge = cv2.filter2D(img_blur2, -1, edge_detection_kernel)
    #img_cont = img_dilate

    #img_cont = (img_dilate + img_thresh) // 2
    #img_cont = (img_edge * 2 + img_thresh) // 3

    #img_cont = cv2.threshold(img_gray, 50, 100, cv2.THRESH_TOZERO)[1]
    #img_cont = cv2.equalizeHist(img_thresh)
    #img_cont = clahe.apply(img_cont)
    #img_cont = cv2.GaussianBlur(img_cont, (5, 5), 0)
    #img_cont = cv2.threshold(img_cont, 50, 100, cv2.THRESH_TOZERO[1])

    # Set up parameters for blob detector
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 40
    params.maxThreshold = 255
    params.filterByColor = False
    params.filterByArea = False
    params.filterByCircularity = True
    params.minCircularity = 0.4
    params.filterByConvexity = True
    params.minConvexity = 0.8
    params.filterByInertia = True
    params.minInertiaRatio = 0.05
    #params.maxInertiaRatio = 0.7

    params2 = cv2.SimpleBlobDetector_Params()
    params2.minThreshold = brightness_pyramid[-1]
    params2.maxThreshold = 255
    params2.filterByColor = False
    params2.filterByArea = False
    params2.filterByCircularity = True
    params2.minCircularity = 0.4
    params2.filterByConvexity = True
    params2.minConvexity = 0.8
    params2.filterByInertia = True
    params2.minInertiaRatio = 0.05
    #params2.maxInertiaRatio = 0.7

    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3 :
        detector = cv2.SimpleBlobDetector(params)
    else :
        detector = cv2.SimpleBlobDetector_create(params)


    # Detect blobs.
    keypoints = detector.detect(img_thresh)
    keypoints_contrast = detector.detect(img_cont)
    if len(keypoints) == 0:
        print('no keypoints found!')

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
    # the size of the circle corresponds to the size of blob

    img_with_keypoints = cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2RGB)
    #img_with_keypoints = cv2.drawKeypoints(img_thresh, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    for keypt in keypoints:
        pt = Coord(int(keypt.pt[0]), int(keypt.pt[1]))
        size = int(keypt.size)
        img_with_keypoints = cv2.circle(img_with_keypoints, (pt.x, pt.y), size, (0, 0, 255), 1)
        #img_with_keypoints = cv2.circle(img_with_keypoints, (pt.x, pt.y), 3, (0, 0, 255), -1)
        img_with_keypoints = cv2.line(img_with_keypoints, (pt.x, pt.y + size), (pt.x, pt.y - size), (0, 0, 255), 1)
        img_with_keypoints = cv2.line(img_with_keypoints, (pt.x + size, pt.y), (pt.x - size, pt.y), (0, 0, 255), 1)
        img_with_keypoints = cv2.putText(img_with_keypoints, '1', (pt.x + size, pt.y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 1)
    #img_with_keypoints= cv2.drawKeypoints(img_with_keypoints, keypoints, np.array([]), (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    for keypt in keypoints_contrast:
        pt = Coord(int(keypt.pt[0]), int(keypt.pt[1]))
        size = int(keypt.size)
        img_with_keypoints = cv2.circle(img_with_keypoints, (pt.x, pt.y), size, (0, 255, 0), 1)
        #img_with_keypoints = cv2.circle(img_with_keypoints, (pt.x, pt.y), 3, (0, 255, 0), -1)
        img_with_keypoints = cv2.line(img_with_keypoints, (pt.x, pt.y + size), (pt.x, pt.y - size), (0, 255, 0), 1)
        img_with_keypoints = cv2.line(img_with_keypoints, (pt.x + size, pt.y), (pt.x - size, pt.y), (0, 255, 0), 1)
        img_with_keypoints = cv2.putText(img_with_keypoints, '2', (pt.x - size * 2, pt.y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 1)

    for i in range(len(avg_row_ind)):
        img_with_keypoints = cv2.circle(img_with_keypoints, (avg_row_ind[i], avg_col_ind[i]), 3, (255, 0, 0), 1)

    img_with_keypoints = np.hstack((img, cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2RGB), cv2.cvtColor(img_cont, cv2.COLOR_GRAY2RGB), img_with_keypoints))
    cv2.imshow("Keypoints", img_with_keypoints)
    cv2.waitKey(0)


def calc_hist_percentile(img, min_bright, percentile):
    hist = cv2.calcHist([img], [0], None, [256 - min_bright], [min_bright, 256])
    sample_size = sum(hist)
    thresh_num_pixel = sample_size * percentile / 100
    num_counted = 0
    for idx in range(len(hist)):
        num_pixel = hist[idx][0]
        if num_counted >= thresh_num_pixel:
            print('%fth percentile: %d' % (percentile, idx + min_bright))
            #plt.plot(hist)
            #plt.show()
            return idx + min_bright
        num_counted += num_pixel
    raise ValueError('Percentile not found')

def find_bright_spots(img, size_gaussian, threshold, nErosion, nDilation):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel_size = size_gaussian
    current_threshold = threshold
    max_pixel = 0
    while max_pixel <= current_threshold:
        if kernel_size < 0:
            kernel_size = size_gaussian
            current_threshold = int(current_threshold * 0.8)
            print('threshold lowered to %d' % current_threshold)
        img_blur = cv2.GaussianBlur(img_gray, (kernel_size, kernel_size), 0)
        max_pixel = img_blur.max()
        print('max pixel for gaussian kernel size %d: %d' % (kernel_size, max_pixel))
        kernel_size -= 2
    img_thresh = cv2.threshold(img_blur, current_threshold, 255, cv2.THRESH_BINARY)[1]
    #filtered = cv2.erode(filtered, None, iterations=nErosion)
    #filtered = cv2.dilate(filtered, None, iterations=nDilation)
    #resized = cv2.resize(img_thresh, (720, 480))
    cv2.imshow('filtered', img_thresh)
    return


def main(argv):
    img_dir = os.path.abspath(argv[1])
    img_list = [x for x in os.listdir(img_dir) if x.endswith('.jpg') or x.endswith('.png')]
    print(img_list)
    for img_name in img_list:
        print(img_name)
        img = cv2.imread(os.path.join(img_dir, img_name))
        #resized = cv2.resize(img, (720, 480))
        #cv2.imshow('original', img)
        #find_bright_spots(img, 11, 60, 1, 2)
        detect_blob(img)
        #cv2.waitKey(0)
    return


if __name__ == '__main__':
    main(sys.argv)
