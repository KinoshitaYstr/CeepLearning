import numpy as np
from PIL import Image
import csv
import glob

def saveImageFromNumpy(m,fname):
    img = Image.fromarray(np.uint8(m))
    img.save(fname)

def saveImageFromCSV(csv_name,fname):
    f = open(csv_name,"r")
    data_list = []
    for line in f:
        l = []
        for val in line.split(","):
            l.append(int(val))
        data_list.append(np.array(l))
    data = np.array(data_list)
    saveImageFromNumpy(data,fname)
    f.close()


if __name__ == "__main__":
    fnames = glob.glob("result/*.csv")
    for fname in fnames:
        fname2 = "result_img/{0}".format(fname.split("/")[-1])
        fname2 = "{0}.jpg".format(fname2.split(".")[0])
        print("{0} -> {1}".format(fname,fname2))
        saveImageFromCSV(fname,fname2)