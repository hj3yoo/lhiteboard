import io
import time
import threading
import picamera
import cv2
import numpy as np
import sys
from io import BytesIO
import debug_render as dbr
import calib_save as cs

import detect

NUM_THREADS = 8 

result_lock = threading.Lock()
result_table = {}
dr = dbr.DebugRenderer()

#calib_coords gets populated during calibration
#stored in this order: L0(x=0,y=0),L1(0,1), L2(1,1), L3(1,0)
#algorithm from: https://math.stackexchange.com/questions/13404/mapping-irregular-quadrilateral-to-a-rectangle/1361366#1361366
#calib_coords = []
warp_matrix = None

def to_normalized_screen_coords(raw_coord):
    np_raw = np.float32([raw_coord[0], raw_coord[1], 1.0])
    warped = warp_matrix.dot(np_raw)
    return (warped[0]/warped[2]/640.0, warped[1]/warped[2]/480.0)

class Consumer(threading.Thread):
    def __init__(self):
        super(Consumer, self).__init__()
        self.terminated = False
        self.i = 0
        self.start()

    def run(self):
        print("Consumer running still")
        while not self.terminated:
            with result_lock:
                get = result_table.get(self.i)
                if get is not None:
                    print("Consumed coordinate {0}: {1}".format(self.i, get))
                    if get != (-1, -1):
                        nsc = to_normalized_screen_coords(get)
                        print("---> NSC: {0}".format(nsc))
                        dr.push_point_mt(nsc[0], nsc[1])
                    self.i += 1
                    
class ImageProcessor(threading.Thread):
    def __init__(self, owner):
        super(ImageProcessor, self).__init__()
        self.stream = io.BytesIO()
        self.event = threading.Event()
        self.terminated = False
        self.owner = owner
        self.start()

    def run(self):
        # This method runs in a separate thread
        while not self.terminated:
            # Wait for an image to be written to the stream
            if self.event.wait(1):
                try:
                    self.stream.seek(0)
                    # Read the image and do some processing on it
                    #Image.open(self.stream)
                    with self.owner.lock:
                        idx = self.owner.frames_processed
                        self.owner.frames_processed += 1

                    image = cv2.imdecode(np.fromstring(self.stream.getvalue(), 
                        dtype=np.uint8), cv2.IMREAD_COLOR)

                    sources, _ = detect.find_source(image)

                    if len(sources) != 0:
                        (x0, y0), (x1, y1) = sources[0]
                        coord, _ = detect.find_coordinate(image[y0:y1, x0:x1])
                        coord = (coord[0] + x0, coord[1] + y0)
                    else:
                        coord = (-1, -1)
                    #print("Found coordinate for frame {0}: {1}".format(idx, coord))

                    if coord != (-1, -1):
                        with self.owner.lock:
                            self.owner.frames_detected += 1

                    with result_lock:
                        result_table[idx] = coord

                    #...
                    #...
                    # Set done to True if you want the script to terminate
                    # at some point
                    #self.owner.done=True
                finally:
                    # Reset the stream and event
                    self.stream.seek(0)
                    self.stream.truncate()
                    self.event.clear()
                    # Return ourselves to the available pool
                    with self.owner.lock:
                        self.owner.pool.append(self)

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
                    #print("WARNING: dropped frame")
                    #print()
                    self.processor = None
        if self.processor:
            self.processor.stream.write(buf)

    def flush(self):
        # When told to flush (this indicates end of recording), shut
        # down in an orderly fashion. First, add the current processor
        # back to the pool
        if self.processor:
            with self.lock:
                self.pool.append(self.processor)
                self.processor = None
        # Now, empty the pool, joining each thread as we go
        while True:
            with self.lock:
                try:
                    proc = self.pool.pop()
                except IndexError:
                    pass # pool is empty
            proc.terminated = True
            proc.join()

# sensor_mode 6 boosts the FPS
# read more about the camera modes here: https://picamera.readthedocs.io/en/release-1.13/fov.html#camera-modes

class CameraThread(threading.Thread):
    def __init__(self, camera, output):
        super(CameraThread, self).__init__()
        self.camera = camera
        self.output = output
        self.start()

    def run(self):
        camera.start_recording(output, format='mjpeg')
        while not output.done:
            camera.wait_recording(1)
        camera.stop_recording()

def calibrate(camera):
    calib_coords = []
    # Calibration - let the user grab 4 coordinates 
    # U press keyboard to take pic
    dirs = ["TOP LEFT", "TOP RIGHT", "BOT RIGHT", "BOT LEFT"]
    for i in range(len(dirs)):
        print("Taking {0} calibration picture. Press keyboard when ready.".format(dirs[i]))
        #camera.start_preview()
        sys.stdin.readline()
        stream = BytesIO() 
        camera.capture(stream, format='jpeg')
        stream.seek(0)
        image = cv2.imdecode(np.fromstring(stream.getvalue(), dtype=np.uint8), 
            cv2.IMREAD_COLOR)
        sources, _ = detect.find_source(image)

        if len(sources) != 0:
            print(sources)
            (x0, y0), (x1, y1) = sources[0]
            coord, _ = detect.find_coordinate(image[y0:y1, x0:x1])
            print(coord)
            coord = (coord[0] + x0, coord[1] + y0)
        else:
            sys.exit("Re-calibration necessary!")
        
        calib_coords.append(coord)
        return calib_coords
        #camera.stop_preview()

with picamera.PiCamera(sensor_mode=5) as camera:
    # Capture grayscale image instead of colour
    camera.color_effects = (128, 128)

    #camera.start_preview() #This outputs the video full-screen in real time
    time.sleep(2)

    answer = input("Do you want to use the saved calibration data? [y/n]:")
    if answer == "y" or answer == "Y":
        calib_coords = cs.read_calib()
    else:
        dr.show_calib_img()
        calib_coords = calibrate(camera)
        answer = input("Save this calibration data? [y/n]:")
        if answer == "y" or answer == "Y":
            cs.save_calib(calib_coords)

    print("Calibration coordinates: {0}".format(calib_coords))
    np_calib_points = np.float32([
        [calib_coords[0][0], calib_coords[0][1]], 
        [calib_coords[1][0], calib_coords[1][1]], 
        [calib_coords[2][0], calib_coords[2][1]], 
        [calib_coords[3][0], calib_coords[3][1]]
    ])
    np_warped_points = np.float32([[dbr.CALIB_BORDER, dbr.CALIB_BORDER], 
        [640-dbr.CALIB_BORDER, dbr.CALIB_BORDER], 
        [640-dbr.CALIB_BORDER, 480-dbr.CALIB_BORDER], 
        [dbr.CALIB_BORDER, 480-dbr.CALIB_BORDER]])
    warp_matrix = cv2.getPerspectiveTransform(np_calib_points, np_warped_points)

    # actually start our threads now    
    output = ProcessOutput()
    consumer = Consumer()
    time_begin = time.time()
    dr.show_clear()

    import signal, sys
    def signal_handler(signal, frame):
        time_now = time.time()
        fps = output.frames_processed / (time_now - time_begin)
        dropped_percent = output.frames_dropped / (output.frames_dropped
            + output.frames_processed) * 100.0
        detected_percent = output.frames_detected / output.frames_processed * 100.0;
        print("Average FPS thus far: {0}".format(fps))  
        print("Avg. % of frames dropped: {0}".format(dropped_percent))
        print("Detected percent: {0}".format(detected_percent))
    signal.signal(signal.SIGQUIT, signal_handler)

    cam_thread = CameraThread(camera, output)
    dr.mainloop() 

    # TODO actual cleanup somehow
    print("Quitting...")



