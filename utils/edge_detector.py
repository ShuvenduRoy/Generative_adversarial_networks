import numpy as np
import cv2 as cv
import glob
from matplotlib import pyplot as plt
import os
from tqdm import tqdm

files = sorted(glob.glob(os.path.join('E:\\Datasets\\edge2shoes\\A') + '/*.*'))
print("Total files to process: "+str(len(files)))

for i in tqdm(range(len(files))):
    image = files[i]
    img = cv.imread(image,0)
    edges = cv.Canny(img,128,128)
    edges = cv.bitwise_not(edges)

    name = image.split("\\")[-1]
    cv.imwrite('E:\\Datasets\\edge2shoes\\B\\'+'%s.jpg'%name, edges)
