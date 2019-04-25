import numpy as np
from PIL import Image
import csv

def saveImageFromNumpy(m,fname):
    img = Image.fromarray(np.uint8(m))
    img.save(fname)

def saveImageFromCSV(csv_name,fname):
    f = open(csv_name,"r")
    data_list = []
    for line in f:
        l = []
        for val in line.split(","):
            print(int(val))
            l.append(int(val))
        data_list.append(np.array(l))
    data = np.array(data_list)
    print(data)
    saveImageFromNumpy(data,fname)
    f.close()


if __name__ == "__main__":
    saveImageFromCSV("test.csv","aaa.jpg")

