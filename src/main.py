import time
import picamera
import numpy as np
import cv2
import sys

import detect


RESOLUTION_X = 640
RESOLUTION_Y = 480
FRAMERATE    = 24


def try_readline():
    import select
    while sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.readline()
    else:
        return None

def time_ms():
    import time
    return int(round(time.time() * 1000))


with picamera.PiCamera(sensor_mode=6) as camera:

    camera.resolution = (RESOLUTION_X, RESOLUTION_Y)
    camera.framerate = FRAMERATE
    print("Camera initialized. Starting capture...")

    while True:

        frame_start = time_ms()

        image = np.empty((RESOLUTION_X * RESOLUTION_Y * 3,), dtype=np.uint8)
        camera.capture(image, 'bgr')
        image = image.reshape((RESOLUTION_Y, RESOLUTION_X, 3))
        coord = detect.find_coordinate(image)

        frame_end = time_ms()
        print("Found coordinate: {0} [frame time: {1} ms]".format(coord, frame_end - frame_start))
        #print("frame time: {0} ms".format(frame_end - frame_start))

        if try_readline() is not None:
            break

    print("Quitting...")

