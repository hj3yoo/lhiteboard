from pymouse import PyMouse
import time
import traceback
import threading
import queue
import _thread

MOUSE_UP = 0
MOUSE_DOWN = 1

MOUSE_LEFT = 1
MOUSE_RIGHT = 2

class MouseThread(threading.Thread):

    def __init__(self, mouse_out):
        super(MouseThread, self).__init__()
        self.__mouse = mouse_out
        self.queue = queue.Queue()
        self.start()

    def run(self):
        while True:
            try:
                while True:
                    coord = self.queue.get_nowait()
                    if coord is not None:
                        self.__mouse.process_tick(coord[0], coord[1])
            except queue.Empty:
                pass
            #except Exception as e:
            #    print(e)
            time.sleep(0.01)


class Mouse(PyMouse):
    def __init__(self, drop_tolerance, right_click_thresh):
        super().__init__()
        self.__is_pressed = False
        self.__pressed_duration = 0
        self.__continuous_dropped_frames = 0
        self.__drag_thresh = drop_tolerance
        self.__rc_thresh = right_click_thresh


    def mouse_move(self, norm_screen_x, norm_screen_y):
        #try:
        self.move(int(norm_screen_x * self.screen_size()[0]), int(norm_screen_y * self.screen_size()[1]))
        #except Exception as e: 
        #    print(e)

    def mouse_click(self, norm_screen_x, norm_screen_y, click_type):
        """
        clicktype 1:  left
        clicktype 2:  right
        """
        #try:
        self.click( int(norm_screen_x * self.screen_size()[0]), int(norm_screen_y * self.screen_size()[1]), click_type)
        #except Exception as e: 
        #    print(e)

    def mouse_press(self, norm_screen_x, norm_screen_y, click_type):
        self.press(int(norm_screen_x * self.screen_size()[0]), int(norm_screen_y * self.screen_size()[1]), click_type)

    def mouse_release(self, norm_screen_x, norm_screen_y, click_type):
        self.release(int(norm_screen_x * self.screen_size()[0]), int(norm_screen_y * self.screen_size()[1]), click_type)

    def mouse_drag(self, norm_screen_x, norm_screen_y):
        self.drag(int(norm_screen_x * self.screen_size()[0]), int(norm_screen_y * self.screen_size()[1]))

    def process_tick(self, norm_screen_x, norm_screen_y):
        if norm_screen_x == -1 and norm_screen_y == -1:
            # No coordinate has been found
            if self.__is_pressed and self.__continuous_dropped_frames >= self.__drag_thresh:
                current_pos = self.position()
                self.release(current_pos[0], current_pos[1], MOUSE_LEFT)
            self.__is_pressed = False
            self.__pressed_duration = 0
            self.__continuous_dropped_frames += 1
        else:
            if self.__is_pressed:
                self.mouse_move(norm_screen_x, norm_screen_y)
            else:
                self.mouse_press(norm_screen_x, norm_screen_y, MOUSE_LEFT)
            self.__is_pressed = True
            self.__continuous_dropped_frames = 0
            self.__pressed_duration += 1

if __name__ == "__main__":
    mouse = Mouse(drop_tolerance=4, right_click_thresh=30)
    thread = MouseThread(mouse)
    #mouse_click(0.5, 0.5, 1)

