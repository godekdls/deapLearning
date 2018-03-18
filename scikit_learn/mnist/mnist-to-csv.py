import struct

def to_csv(name, maxdata):
    # open files
    label_file = open("./sample/" + name + "-labels-idx1-ubyte", "rb")
    image_file = open("./sample/" + name + "-images-idx3-ubyte", "rb")
    csv_file = open("./sample/" + name + ".csv", "w", encoding="utf-8")

    # read header
    mag, label_count = struct.unpack(">II", label_file.read(8)) # read 8 byte and cast to integer
    mag, image_count = struct.unpack(">II", image_file.read(8))
    rows, cols = struct.unpack(">II", image_file.read(8))
    pixels = rows * cols

    # read image data and save into csv
    for idx in range(label_count):
        if idx > maxdata: break
        label = struct.unpack("B", label_file.read(1))[0]
        binary_data = image_file.read(pixels)
        string_data =  list(map(lambda n: str(n), binary_data))
        csv_file.write(str(label) + ",")
        csv_file.write(",".join(string_data) + "\r\n")

    csv_file.close()
    label_file.close()
    image_file.close()

to_csv("train", 1000)
to_csv("t10k", 500)