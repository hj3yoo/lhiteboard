from pymouse import PyMouse
import time
import threading
import queue
from itertools import combinations

MOUSE_LEFT = 1
MOUSE_RIGHT = 2


def square_dist(x, y):
    return sum([(xi - yi) ** 2 for xi, yi in zip(x, y)]) ** (1 / 2)


def calc_max_dist(coords):
    max_dist = 0
    for left, right in combinations(coords, 2):
        max_dist = max(max_dist, square_dist(left, right))
    return max_dist


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
            time.sleep(0.01)


class Mouse(PyMouse):
    def __init__(self, drop_tolerance, right_click_dist, right_click_duration):
        super().__init__()
        self.__is_pressed = False
        self.__continuous_dropped_frames = 0
        self.__continuous_coords = []
        self.__drag_thresh = drop_tolerance  # How many frames can I skip until the click breaks?
        self.__rc_duration = right_click_duration  # How long should I be holding to register right click?
        self.__rc_dist = right_click_dist  # How far can I move before right click fails to register?

    def process_tick(self, norm_screen_x, norm_screen_y):
        x = int(norm_screen_x * self.screen_size()[0])
        y = int(norm_screen_y * self.screen_size()[1])
        if norm_screen_x == -1 and norm_screen_y == -1:
            # No coordinate has been found - pen released
            # If there are no input for several frames, assume the pen was released
            if self.__is_pressed and self.__continuous_dropped_frames >= self.__drag_thresh:
                current_pos = self.position()
                max_dist = calc_max_dist(self.__continuous_coords)
                if len(self.__continuous_coords) >= self.__rc_duration and max_dist < self.__rc_dist:
                    # This is a right click - pen was held in a small area for a fixed duration
                    self.click(*current_pos, button=MOUSE_RIGHT)
                else:
                    self.release(*current_pos, button=MOUSE_LEFT)
                # Reset
                self.__is_pressed = False
                self.__continuous_coords = []
            self.__continuous_dropped_frames += 1
        else:
            if self.__is_pressed:
                self.move(x, y)
            else:
                self.press(x, y, button=MOUSE_LEFT)
                # Reset
                self.__is_pressed = True
                self.__continuous_dropped_frames = 0
            self.__continuous_coords.append((x, y))


if __name__ == "__main__":
    mouse = Mouse(drop_tolerance=2, right_click_duration=30, right_click_dist=15)
    thread = MouseThread(mouse)
