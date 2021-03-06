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
from scipy.ndimage.measurements import label

dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
svc = dist_pickle["svc"]
X_scaler = dist_pickle["X_scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]

files = glob.glob("./test_images/*.jpg")
image_name = "test"
count = 0

for file in files:
    image = mpimg.imread(file)
    count = count + 1
    get_filename = lambda name: image_name + "_" + str(count) + "_" + name
    draw_image = np.copy(image)

    ystart = 400
    ystop = 656
    scale = 1.5

    out_img, box_list = find_cars(draw_image,
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

    # Heat Map
    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat, box_list)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,1)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    fig = plt.figure()

    # plt.subplot('131')
    plt.imshow(out_img)
    plt.title('find cars')
    plt.savefig(get_filename("find_cars.png"));

    fig = plt.figure()
    # plt.subplot('132')
    plt.imshow(draw_img)
    plt.title('Car Positions')
    plt.savefig(get_filename("car_positions.png"));

    fig = plt.figure()
    # plt.subplot('133')
    plt.imshow(heatmap, cmap='hot')
    plt.title('Heat Map')
    # fig.tight_layout()

    plt.savefig(get_filename("heat_map.png"));
