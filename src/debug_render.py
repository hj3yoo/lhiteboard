from tkinter import *
import queue
import _thread
import time

POINT_WIDTH=10
POINT_HEIGHT=10
UPDATE_DELAY_SEC=(1.0/30)

CALIB_BORDER = 0

class DebugRenderer():
    def __init__(self):
        self.master = Tk()
        self.master.attributes("-fullscreen", True)
        self.w = self.master.winfo_screenwidth()
        self.h = self.master.winfo_screenheight()
        self.canvas = Canvas(self.master, width=self.w, height=self.h)
        self.canvas.config(background="teal")
        self.canvas.pack()
        self.queue = queue.Queue()
        self.cp = []

    def show_calib_img(self):
        radius = 100
        self.canvas.create_oval(
            CALIB_BORDER - radius, 
            CALIB_BORDER - radius, 
            CALIB_BORDER + radius, 
            CALIB_BORDER + radius)
        self.canvas.create_oval(
            self.w - CALIB_BORDER - radius, 
            CALIB_BORDER - radius, 
            self.w - CALIB_BORDER + radius, 
            CALIB_BORDER + radius)
        self.canvas.create_oval(self.w - CALIB_BORDER - radius, 
            self.h - CALIB_BORDER - radius, 
            self.w - CALIB_BORDER + radius, 
            self.h - CALIB_BORDER + radius)
        self.canvas.create_oval(
            CALIB_BORDER - radius, 
            self.h - CALIB_BORDER - radius, 
            CALIB_BORDER + radius, 
            self.h - CALIB_BORDER + radius)
        self.master.update()

    def push_point_mt(self, x, y):
        """
        This is the only method that can be called from a thread other than
        the main thread.
        """
        self.queue.put((x, y))

    def mainloop(self):
        while True:
            self.mainloop_tick()

    def mainloop_tick(self):
        try:
            while True:
                coord = self.queue.get_nowait()
                if coord is not None:
                    self.show_point(coord[0], coord[1])
        except queue.Empty:
            pass
        time.sleep(UPDATE_DELAY_SEC)


    def show_point(self, x, y, radius=POINT_WIDTH, transform=True):
        """
        Input is the coordinate found by the image detection algorithm.
        (-1, -1) if it is not valid and (0, 0) being in the top left corner.
        """
        if transform:
            coord = self.normalized_cam_to_canvas(x, y)
        else:
            coord = (x, y)
            
        if x != -1 and y != -1:
            self.canvas.create_oval(
                coord[0] - radius,
                coord[1] - radius,
                coord[0] + radius,
                coord[1] + radius
            )

        self.master.update()

    def show_clear(self):
        self.canvas.delete("all")
        self.master.update()

    def normalized_cam_to_canvas(self, ncx, ncy):
        return (self.w * ncx, self.h * ncy)


if __name__ == "__main__":
    dbr = DebugRenderer(640, 480) 
    while True:
        dbr.push_point_mt(0.5, 0.5);
        dbr.mainloop_tick()

