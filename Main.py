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
from moviepy.editor import VideoFileClip

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
        self.y_start = 400
        self.y_stop = 650


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


def convert_color(img, cfg: Config):
    if cfg.color_space == 'HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif cfg.color_space == 'LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    elif cfg.color_space == 'HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    elif cfg.color_space == 'YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    elif cfg.color_space == 'YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        return np.copy(img)


class Train:

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def img_features(self, img):
        features = []
        cfg = self.cfg

        feature_image = convert_color(img, cfg)

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

    def imgs_features(self, files):
        features = []
        for file in files:
            img = mpimg.imread(file)
            features1 = self.img_features(img)
            features.append(features1)

        return features

    def prepare_all_features(self, sample_size=0):
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

        car_features = self.imgs_features(cars)
        noncar_features = self.imgs_features(notcars)

        logging.info('%.2f Seconds to prepare features', time.time() - t)

        return car_features, noncar_features

    def train_svc(self):
        car_features, notcar_features = self.prepare_all_features(sample_size=0)
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
            pickle.dump({'svc': svc, 'scaler': X_scaler}, f)

        return svc


def load_svc():
    if os.path.isfile('svc.pickle'):
        with open('svc.pickle', 'rb') as f:
            return pickle.load(f)


class FindCars:

    def __init__(self, cfg: Config, svc, X_scaler, scale):
        self.cfg = cfg
        self.svc = svc
        self.X_scaler = X_scaler
        self.scale = scale

    def find_cars(self, img):
        draw_img = np.copy(img)
        #img = img.astype(np.float32) / 255
        cfg = self.cfg

        img_tosearch = img[cfg.y_start:cfg.y_stop, :, :]
        ctrans_tosearch = convert_color(img_tosearch, cfg)
        if self.scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch,
                                (np.int(imshape[1] / self.scale), np.int(imshape[0] / self.scale)))

        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]

        nxblocks = (ch1.shape[1] // cfg.pix_per_cell) - cfg.cell_per_block + 1
        nyblocks = (ch1.shape[0] // cfg.pix_per_cell) - cfg.cell_per_block + 1
        nfeat_per_block = cfg.orient * cfg.cell_per_block ** 2

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // cfg.pix_per_cell) - cfg.cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        hogs = []
        if cfg.hog_channel == 'ALL':
            hogs.append(get_hog_features(ch1, cfg.orient, cfg.pix_per_cell, cfg.cell_per_block, feature_vec=False))
            hogs.append(get_hog_features(ch2, cfg.orient, cfg.pix_per_cell, cfg.cell_per_block, feature_vec=False))
            hogs.append(get_hog_features(ch3, cfg.orient, cfg.pix_per_cell, cfg.cell_per_block, feature_vec=False))
        else:
            hogs.append(get_hog_features(ctrans_tosearch[:, :, cfg.hog_channel], cfg.orient, cfg.pix_per_cell, cfg.cell_per_block, feature_vec=False))

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step

                if cfg.hog_channel == 'ALL':
                    hog_feat1 = hogs[0][ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    hog_feat2 = hogs[0][ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    hog_feat3 = hogs[0][ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                else:
                    hog_features = hogs[0][ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()

                xleft = xpos * cfg.pix_per_cell
                ytop = ypos * cfg.pix_per_cell

                subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

                spatial_features = bin_spatial(subimg, size=cfg.spatial_size)
                hist_features = color_hist(subimg, nbins=cfg.hist_bins)

                features = np.hstack((spatial_features, hist_features, hog_features))
                features = features.reshape(1, -1)
                test_features = self.X_scaler.transform(features)
                is_car = self.svc.predict(test_features)

                if is_car == 1:
                    xbox_left = np.int(xleft * self.scale)
                    ytop_draw = np.int(ytop * self.scale)
                    win_draw = np.int(window * self.scale)
                    cv2.rectangle(draw_img, (xbox_left, ytop_draw + cfg.y_start),
                                  (xbox_left + win_draw, ytop_draw + win_draw + cfg.y_start),
                                  (0, 0, 255), 6)

        return draw_img


def main():

    cfg = Config()
    train = Train(cfg)
    svc = train.train_svc()



if __name__ == '__main__':
    main()