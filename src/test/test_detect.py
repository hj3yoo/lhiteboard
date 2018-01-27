from src.detect import *
import argparse
import os
import sys
import math

def detection_test(arg_dict):
    """
    Test the point detection algorithm using a given test data
    :param arg_dict: dictionary of all arguments parsed using argparser
    """
    img_dir = os.path.abspath(arg_dict['img_dir'])
    img_list = [x for x in os.listdir(img_dir) if x.endswith('.jpg') or x.endswith('.png')]
    print('Number of test images: %d' % len(img_list))
    """
    Coordinate detection can be computationally heavy for the entire image
    Therefore, image is cropped into a smaller sections where the IR reflection is detected
    For each cropped section, coordinate detection will be run
    """
    for img_name in img_list:
        print(img_name + ':')
        img = cv2.imread(os.path.join(img_dir, img_name))
        img_show = copy.copy(img)
        sources = find_source(img)
        new_x0 = math.inf
        new_y0 = math.inf
        new_x1 = -1
        new_y1 = -1
        # Each sections are represented by a rectangle
        for pt0, pt1 in sources:
            x0, y0 = pt0
            x1, y1 = pt1
            det_pt = find_coordinate(img[y0:y1, x0:x1])
            if det_pt[0] != -1 and det_pt[1] != -1:
                # Mark the detected point for each section
                img_show = cv2.circle(img_show, (x0 + det_pt[0], y0 + det_pt[1]), arg_dict['pt_radius'], COLOR_RED, 3)
            img_show = cv2.rectangle(img_show, pt0, pt1, COLOR_GREEN, 3)
            new_x0 = min(new_x0, x0)
            new_y0 = min(new_y0, y0)
            new_x1 = max(new_x1, x1)
            new_y1 = max(new_y1, y1)

        # Display the result to the user, and pause until user proceeds
        #cv2.imshow('Result zoomed', img_show[new_y0-10:new_y1+10, new_x0-10:new_x1+10])
        img_show = cv2.resize(img_show, (len(img[0]) // 4, len(img) // 4))
        cv2.imshow('Result', img_show)
        if not arg_dict['animate']:
            cv2.waitKey(0)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', required=True, dest='img_dir', type=str, help='Path of the image source folder')
    parser.add_argument('-ani', dest='animate', action='store_true',
                        help='Quickly animate through all test images instead of manual inspection')
    parser.add_argument('-ptr', dest='pt_radius', type=int, default=10,
                        help='Radius of the circles to mark input points')
    args = parser.parse_args()
    dict_args = vars(args)
    in_dir = os.path.abspath(dict_args['img_dir'])
    try:
        if len(os.listdir(in_dir)) == 0:
            print('The specified directory is empty')
            sys.exit()
    except FileNotFoundError:
        print(in_dir)
        print('The specified directory does not exist')
        sys.exit()
    detection_test(dict_args)
