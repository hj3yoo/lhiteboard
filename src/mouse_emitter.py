from pymouse import PyMouse

m = None

def init():
    global m
    m = PyMouse()
    res = m.screen_size()

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
    m.click( int(norm_screen_x * m.screen_size()[0]), int(norm_screen_y * m.screen_size()[1]), clicktype )

if __name__ == "__main__":
    init()
    mouse_click(0.5, 0.5, 1)

