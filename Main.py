import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from skimage.feature import hog
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os.path

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S', level=logging.INFO)
logging.info('Vehicule detection initializing')


class Config:

    def __init__(self):
        self.color_space = 'LUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = 9  # HOG orientations
        self.pix_per_cell = 8  # HOG pixels per cell
        self.cell_per_block = 2  # HOG cells per block
        self.hog_channel = 0  # Can be 0, 1, 2, or "ALL"
        self.spatial_size = (16, 16)  # Spatial binning dimensions
        self.hist_bins = 16  # Number of histogram bins
        self.spatial_feat = True  # Spatial features on or off
        self.hist_feat = True  # Histogram features on or off
        self.hog_feat = True  # HOG features on or off
        self.y_start_stop = [400, 650]  # Min and max in y to search in slide_window()


def bin_spatial(img, size=(32, 32)):
    features = cv2.resize(img, size).ravel()
    return features


# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    return np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):

    if vis:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec,
                                  block_norm='L2-Hys')
        return features, hog_image
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec,
                       block_norm='L2-Hys')
        return features


def img_features(img, cfg: Config):
    features = []

    if cfg.color_space == 'HSV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif cfg.color_space == 'LUV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    elif cfg.color_space == 'HLS':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    elif cfg.color_space == 'YUV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    elif cfg.color_space == 'YCrCb':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)

    if cfg.spatial_feat:
        spatial_features = bin_spatial(feature_image, size=cfg.spatial_size)
        features.append(spatial_features)

    if cfg.hist_feat:
        hist_features = color_hist(feature_image, nbins=cfg.hist_bins)
        features.append(hist_features)

    if cfg.hog_feat:
        if cfg.hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                     cfg.orient, cfg.pix_per_cell, cfg.cell_per_block,
                                                     vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, cfg.hog_channel], cfg.orient,
                                            cfg.pix_per_cell, cfg.cell_per_block,
                                            vis=False, feature_vec=True)
        features.append(hog_features)

    return np.concatenate(features)


def imgs_features(files, cfg: Config):
    features = []
    for file in files:
        img = mpimg.imread(file)
        features1 = img_features(img, cfg)
        features.append(features1)

    return features


def prepare_all_features(cfg: Config, sample_size=0):
    cars = []
    images = glob.glob('vehicles_smallset/*/*.jpeg', recursive=True)
    #images = glob.glob('vehicles/*/*.png', recursive=True)
    for image in images:
        cars.append(image)
    notcars = []
    images = glob.glob('non-vehicles_smallset/*/*.jpeg', recursive=True)
    #images = glob.glob('non-vehicles/*/*.png', recursive=True)
    for image in images:
        notcars.append(image)

    if sample_size > 0:
        cars = cars[0:sample_size]
        notcars = notcars[0:sample_size]

    logging.info('#vehicules: %d #nonVehicules: %d', len(cars),  len(notcars))

    t = time.time()

    car_features = imgs_features(cars, cfg)
    noncar_features = imgs_features(notcars, cfg)

    logging.info('%.2f Seconds to prepare features', time.time() - t)

    return car_features, noncar_features


def train_svc(car_features, notcar_features):
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)

    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    logging.info('Feature vector length: %d', len(X_train[0]))

    svc = LinearSVC()
    t = time.time()
    svc.fit(X_train, y_train)

    logging.info('%.2f Seconds to train SVC', time.time() - t)
    logging.info('Test Accuracy of SVC = %.4f', svc.score(X_test, y_test))

    with open('svc.pickle', 'wb') as f:
        pickle.dump(svc, f)

    return svc


def load_svc():
    if os.path.isfile('svc.pickle'):
        with open('svc.pickle', 'rb') as f:
            return pickle.load(f)




def main():

    cfg = Config()
    car_features, notcar_features = prepare_all_features(cfg, sample_size=0)
    svc = train_svc(car_features, notcar_features)

    # scv = load_svc()


if __name__ == '__main__':
    main()