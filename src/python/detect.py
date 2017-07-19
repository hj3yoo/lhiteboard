import os.path
import cv2
import numpy as np
import sys

class Coord:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return '(%d, %d)' % (self.x, self.y)


def detect_blob(img):
    # Filter the image with Gaussian blur & binary threshold
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_thresh = cv2.threshold(img_blur, 60, 100, cv2.THRESH_TOZERO)[1]

    # Set up parameters for blob detector
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 40
    params.maxThreshold = 255
    params.filterByColor = False
    params.filterByArea = False
    params.filterByCircularity = True
    params.minCircularity = 0.2
    params.filterByConvexity = True
    params.minConvexity = 0.5
    params.filterByInertia = True
    params.minInertiaRatio = 0.05
    #params.maxInertiaRatio = 0.7

    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3 :
        detector = cv2.SimpleBlobDetector(params)
    else :
        detector = cv2.SimpleBlobDetector_create(params)


    # Detect blobs.
    keypoints = detector.detect(img_thresh)
    if len(keypoints) == 0:
        print('no keypoints found!')

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
    # the size of the circle corresponds to the size of blob

    im_with_keypoints = cv2.drawKeypoints(img_thresh, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show blobs
    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.waitKey(0)



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
        cv2.imshow('original', img)
        #find_bright_spots(img, 11, 60, 1, 2)
        detect_blob(img)
        #cv2.waitKey(0)
    return


if __name__ == '__main__':
    main(sys.argv)
