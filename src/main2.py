import io
import threading
import picamera
import cv2
import numpy as np
import debug_render as dbr
import calib_save as cs
import signal
import sys
from mouse_emitter import *
import time
import json
from PIL import Image

import detect


def to_normalized_screen_coords(raw_coord):
    np_raw = np.float32([raw_coord[0], raw_coord[1], 1.0])
    warped = warp_matrix.dot(np_raw)
    return warped[0] / warped[2] / WIDTH, warped[1] / warped[2] / HEIGHT


class Consumer(threading.Thread):
    def __init__(self):
        super(Consumer, self).__init__()
        self.terminated = False
        self.i = 0
        self.start()

    def run(self):
        #print("Consumer running still")
        while not self.terminated:
            with result_lock:
                get = result_table.get(self.i)
                if get is not None:
                    #print("Consumed coordinate {0}: {1}".format(self.i, get))
                    if get != (-1, -1):
                        nsc = to_normalized_screen_coords(get)
                        #print("---> NSC: {0}".format(nsc))
                        #dr.push_point_mt(nsc[0], nsc[1])
                        mouse_thread.queue.put(nsc)
                    else:
                        #dr.push_point_mt(*get)
                        mouse_thread.queue.put(get)
                    self.i += 1
        #print('Consumer terminating')

    def terminate(self):
        self.terminated = True


class ImageProcessor(threading.Thread):
    def __init__(self, owner):
        super(ImageProcessor, self).__init__()
        self.stream = io.BytesIO()
        self.event = threading.Event()
        self.terminated = False
        self.owner = owner
        self.start()

    def run(self):
        #print('ImageProcessor running - Thread #%d' % threading.get_ident())
        # This method runs in a separate thread
        while not self.owner.done:
            # Wait for an image to be written to the stream
            if self.event.wait(1):
                try:
                    self.stream.seek(0)
                    # Read the image and do some processing on it
                    # Image.open(self.stream)
                    with self.owner.lock:
                        idx = self.owner.frames_processed
                        self.owner.frames_processed += 1

                    image = cv2.imdecode(np.fromstring(self.stream.getvalue(),
                                                       dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    # Crop the image to the calibrated area
                    mask = np.zeros_like(image)
                    if len(image.shape) > 2:
                        channel_count = image.shape[2]
                        mask_color = (255,) * channel_count
                    else:
                        mask_color = 255
                    cv2.fillConvexPoly(mask, np_crop_points, mask_color)
                    image_crop = cv2.bitwise_and(image, mask)

                    sources, _ = detect.find_source(image_crop, blur_size=blur_mask_size,
                                                    threshold=source_det_bright_thresh)

                    if len(sources) != 0:
                        (x0, y0), (x1, y1) = sources[0]
                        coord, _ = detect.find_coordinate(image_crop[y0:y1, x0:x1], blur_size=blur_mask_size,
                                                          static_threshold=coord_det_bright_thresh,
                                                          dynamic_threshold=coord_det_bright_percentile)
                        coord = (coord[0] + x0, coord[1] + y0)
                    else:
                        coord = (-1, -1)
                    # print("Found coordinate for frame {0}: {1}".format(idx, coord))

                    if coord != (-1, -1):
                        with self.owner.lock:
                            self.owner.frames_detected += 1

                    with result_lock:
                        result_table[idx] = coord

                        # ...
                        # ...
                        # Set done to True if you want the script to terminate
                        # at some point
                        # self.owner.done=True
                finally:
                    # Reset the stream and event
                    self.stream.seek(0)
                    self.stream.truncate()
                    self.event.clear()
                    # Return ourselves to the available pool
                    with self.owner.lock:
                        self.owner.pool.append(self)
                    time.sleep(0.01)
        #print('ImageProcessor terminating - Thread #%d' % threading.get_ident())


class ProcessOutput(object):
    def __init__(self):
        self.done = False
        self.frames_processed = 0
        self.frames_dropped = 0
        self.frames_detected = 0
        # Construct a pool of 4 image processors along with a lock
        # to control access between threads
        self.lock = threading.Lock()
        self.pool = [ImageProcessor(self) for i in range(NUM_THREADS)]
        self.processor = None

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # New frame; set the current processor going and grab
            # a spare one
            if self.processor:
                self.processor.event.set()
            with self.lock:
                if self.pool:
                    self.processor = self.pool.pop()
                else:
                    # No processor's available, we'll have to skip
                    # this frame; you may want to print a warning
                    # here to see whether you hit this case
                    self.frames_dropped += 1
                    # print("WARNING: dropped frame")
                    # print()
                    self.processor = None
        if self.processor:
            self.processor.stream.write(buf)

    def flush(self):
        # When told to flush (this indicates end of recording), shut
        # down in an orderly fashion. First, add the current processor
        # back to the pool
        self.done = True
        if self.processor:
            with self.lock:
                self.pool.append(self.processor)
                self.processor = None
        # Now, empty the pool, joining each thread as we go
        threads_terminated = 0
        start_ts = time.time()
        elapsed_ts = time.time() - start_ts
        while threads_terminated < NUM_THREADS and elapsed_ts < 3.0:
            with self.lock:
                try:
                    proc = self.pool.pop()
                    proc.terminated = True
                    threads_terminated += 1
                    # proc.join()
                except IndexError:
                    pass  # pool is empty
            elapsed_ts = time.time() - start_ts
        #print('flush finished')
        sys.exit(0)


# sensor_mode 6 boosts the FPS
# read more about the camera modes here: https://picamera.readthedocs.io/en/release-1.13/fov.html#camera-modes
class CameraThread(threading.Thread):
    def __init__(self, camera, output):
        super(CameraThread, self).__init__()
        self.camera = camera
        self.output = output
        self.start()

    def run(self):
        self.camera.start_recording(self.output, format='mjpeg')
        while not self.output.done:
            self.camera.wait_recording(1)
        self.camera.stop_recording()


def calibrate(camera, max_dist_thresh=15):
    dr = dbr.DebugRenderer()

    corner_coords = []
    # Calibration - let the user grab 4 coordinates
    # U press keyboard to take pic
    corners = ["top left", "top right", "bottom right", "bottom left"]
    img_corner_name = ['TL.jpg', 'TR.jpg', 'BR.jpg', 'BL.jpg']
    for i in range(len(corners)):
        corner = corners[i]
        pil_image = Image.open(img_corner_name[i])
        dr.show_img(pil_image)
        num_coord_found = 0
        num_frame_idle = 0
        coords = []
        print('Please point your device towards %s corner of the screen' % corner)
        while True:
            # camera.start_preview()
            #sys.stdin.readline()
            stream = io.BytesIO()
            camera.capture(stream, format='jpeg')
            stream.seek(0)
            image = cv2.imdecode(np.fromstring(stream.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
            sources, _ = detect.find_source(image, blur_size=blur_mask_size, threshold=source_det_bright_thresh)
            if len(sources) != 0:
                (x0, y0), (x1, y1) = sources[0]
                coord, _ = detect.find_coordinate(image[y0:y1, x0:x1])
                if coord != (-1, -1):
                    coord = (coord[0] + x0, coord[1] + y0)
                    coords.append(coord)
                    #print('%s: %s' % (corner, coord))
                    num_coord_found += 1
                    num_frame_idle = 0
                else:
                    num_frame_idle += 1
                    # If nothing has been detected for 20 frames, restart the calibration for this corner
                    if num_frame_idle >= 20 and num_coord_found > 0:
                        print('No input was detected. Please try again.')
                        num_coord_found = 0
                        coords = []

            # No more input source - check all of the detected coordinates are close to each other
            # Repeat this corner if the coordinates are far apart from each other
            elif (len(sources) == 0 or coord == (-1, -1)) and num_coord_found >= 5:
                max_dist = calc_max_dist(coords)
                if max_dist > max_dist_thresh:
                    #print(coords)
                    #print(max_dist)
                    print("The detected coordinates were too far apart. Please try again.")
                    num_coord_found = 0
                    coords = []
                else:
                    # Good to go for next corner
                    print("Successfully calibrated %s" % corner)
                    break
        average_coord = tuple([sum(a) / len(a) for a in zip(*coords)])
        #print('%s average: %s' % (corner, average_coord))
        corner_coords.append(average_coord)
    dr.destroy()
    return corner_coords
        # camera.stop_preview()


if __name__ == '__main__':
    try:
        with picamera.PiCamera(sensor_mode=5) as camera_pi:
            # Capture grayscale image instead of colour
            camera_pi.color_effects = (128, 128)

            # camera_pi.start_preview() #This outputs the video full-screen in real time
            #time.sleep(2)
            with open('settings.json') as settings_file:
                settings = json.load(settings_file)

            NUM_THREADS = settings['NUM_THREAD']

            result_lock = threading.Lock()
            result_table = {}

            # Mouse emulator
            mouse = Mouse(drop_tolerance=settings['DRAG_DROP_THRESH'], right_click_duration=settings['RC_NUM_FRAMES'],
                          right_click_dist=settings['RC_DIST'])
            mouse_thread = MouseThread(mouse)

            WIDTH, HEIGHT = mouse.screen_size()
            blur_mask_size = settings['BLUR_SIZE']
            source_det_bright_thresh = settings['SD_BRIGHT_THRESH']
            coord_det_bright_thresh = settings['CD_BRIGHT_THRESH']
            coord_det_bright_percentile = settings['CD_BRIGHT_PERCENT']
            coord_det_weight = settings['CD_COORD_WEIGHT']


            # Check if the camera was previously calibrated, and ask user if they want recalibration
            is_calibrated = False
            try:
                calib_file = open(cs.CALIB_FILENAME)
                answer = input('The camera has been previously calibrated. Would you like to recalibrate? [y/n]')
                if answer.lower() == 'y':
                    is_calibrated = False
                else:
                    calib_coords = cs.read_calib()
                    is_calibrated = True
            except FileNotFoundError:
                is_calibrated = False

            while not is_calibrated:
                calib_coords = calibrate(camera_pi)
                answer = input("Would you like to recalibrate? [y/n]:")
                if answer.lower() == 'y':
                    is_calibrated = False
                else:
                    cs.save_calib(calib_coords)
                    is_calibrated = True


            #print("Calibration coordinates: {0}".format(calib_coords))
            np_calib_points = np.float32([
                [calib_coords[0][0], calib_coords[0][1]],
                [calib_coords[1][0], calib_coords[1][1]],
                [calib_coords[2][0], calib_coords[2][1]],
                [calib_coords[3][0], calib_coords[3][1]]
            ])
            np_warped_points = np.float32([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]])
            np_crop_points = np.int32(np_calib_points)
            warp_matrix = cv2.getPerspectiveTransform(np_calib_points, np_warped_points)
            #print(warp_matrix)

            # actually start our threads now
            process_output = ProcessOutput()
            consumer = Consumer()
            cam_thread = CameraThread(camera_pi, process_output)
            print('Setting has been finished. You may minimize this terminal now.')

            while True:
                time.sleep(0.001)
            #dr = dbr.DebugRenderer()
            #dr.show_clear()
            #dr.mainloop()

    except KeyboardInterrupt:
        print('Terminating ...')
        process_output.flush()
        consumer.terminate()
        sys.exit(0)
