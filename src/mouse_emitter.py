from pymouse import PyMouse
import time
import traceback
import threading
import queue
import _thread

class MouseState():
    MOUSE_UP   = 0
    MOUSE_DOWN = 1
    
    def __init__(self):
        self.first_detected_ts = None
        self.last_detected = None
        self.last_detected_ts = None
        self.state = MouseState.MOUSE_UP

MOUSE_RIGHT_CLICK_DELAY_THRESH = 3.0
LAST_DETECTED_THRESH = 0.5


class MouseThread(threading.Thread):

    def __init__(self):
        super(MouseThread, self).__init__()
        self.queue = queue.Queue()
        self.start()

    def run(self):
        while True:
            try:
                while True:
                    coord = self.queue.get_nowait()
                    if coord is not None:
                        mouse_tick(coord[0], coord[1])
            except queue.Empty: pass
            except Exception as e:
                print(e)
            time.sleep(0.01) 

m = None
state = MouseState()
thread = None


def init():
    global m
    global thread
    m = PyMouse()
    thread = MouseThread()

def mouse_move(norm_screen_x, norm_screen_y):
    global m
    try:
        m.move( int(norm_screen_x * m.screen_size()[0]), int(norm_screen_y * m.screen_size()[1]) )
    except Exception as e: print(e)

def mouse_click(norm_screen_x, norm_screen_y, clicktype):
    """
    clicktype 1:  left
    clicktype 2:  right
    """
    global m
    try:
        m.click( int(norm_screen_x * m.screen_size()[0]), int(norm_screen_y * m.screen_size()[1]), clicktype )
    except Exception as e: 
        print(e)
        import sys
        sys.exit(-1)

def mouse_tick(norm_screen_x, norm_screen_y):
    global state

    now_ts = time.time()
    detected = (norm_screen_x, norm_screen_y) != (-1, -1)

    if detected:
        state.last_detected = (norm_screen_x, norm_screen_y)
        state.last_detected_ts = now_ts

    if state.state == MouseState.MOUSE_UP:
        # UP -> DOWN transition
        if detected:
            state.first_detected_ts = time.time()
            state.state = MouseState.MOUSE_DOWN

    elif state.state == MouseState.MOUSE_DOWN:
        # DOWN -> UP transition
        if not detected and now_ts - state.last_detected_ts > LAST_DETECTED_THRESH:
            state.state = MouseState.MOUSE_UP 
            if(now_ts - state.first_detected_ts >= MOUSE_RIGHT_CLICK_DELAY_THRESH):
                print("RIGHT CLICK\n\n\n")
                mouse_click(state.last_detected[0], state.last_detected[1], 2)
            else:
                print("LEFT CLICK\n\n\n")
                mouse_click(state.last_detected[0], state.last_detected[1], 1)

if __name__ == "__main__":
    init()
    mouse_click(0.5, 0.5, 1)

