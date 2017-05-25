from Common import *

import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from Config import Config

logging.info('%s initializing', __name__)


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
            pickle.dump({'svc': svc, 'scaler': X_scaler, 'cfg': self.cfg}, f)

        return svc