from subprocess import call

def mouse_move(x, y):
    call(["xdotool", "mousemove", str(x), str(y)])

if __name__ == "__main__":
    mouse_move(20, 20)

