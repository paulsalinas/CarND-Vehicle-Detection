import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from utils import *
import pickle

dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
svc = dist_pickle["svc"]
X_scaler = dist_pickle["X_scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]

image = mpimg.imread('./test_images/test6.jpg')
draw_image = np.copy(image)

ystart = 400
ystop = 656
scale = 1.5

out_img = find_cars(draw_image, 
                    ystart, 
                    ystop, 
                    scale, 
                    svc, 
                    X_scaler, 
                    orient, 
                    pix_per_cell, 
                    cell_per_block, 
                    spatial_size, 
                    hist_bins)

plt.imshow(out_img);
plt.savefig("find_cars.png");


