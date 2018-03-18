CALIB_FILENAME="calib.txt"

def save_calib(calib_coords):
    file = open(CALIB_FILENAME, "w")
    for i in range(0, len(calib_coords)):
        file.write("{0} {1}\n".format(calib_coords[i][0], calib_coords[i][1]))
    file.close()

def read_calib():
    calib_coords = [] 
    file = open(CALIB_FILENAME, "r") 
    lines = file.readlines() 
    lines = [x.strip("\n") for x in lines]
    for line in lines:
        calib_coords.append((
            float(line.split(" ")[0]),
            float(line.split(" ")[1])
        ))
    file.close()
    return calib_coords 

def clear_calib():
    os.remove(CALIB_FILENAME)
     
if __name__ == "__main__":
    calib_coords = [
        (1.2,   2.5 ),
        (2.23,  3.23),
        (2.31,  9.99),
        (1.31,  9.23)
    ]

    save_calib(calib_coords)
    read_back = read_calib()
    clear_calib()
    assert calib_coords == read_back
