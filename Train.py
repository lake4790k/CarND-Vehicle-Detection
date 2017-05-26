from Common import *

import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV

from FeaturesConfig import FeaturesConfig

logging.info('%s initializing', __name__)


class Train:

    def __init__(self, cfg: FeaturesConfig, small=True, sample_size=0, grid=False):
        self.cfg = cfg
        self.small = small
        self.sample_size = sample_size
        self.grid = grid

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
        for file in tqdm(files):
            img = mpimg.imread(file)
            if not self.small:
                img *= 255
                img = img.astype(np.uint8)
            features1 = self.img_features(img)
            features.append(features1)

        return features

    def prepare_all_features(self):
        if self.small:
            cars = glob.glob('vehicles_smallset/*/*.jpeg', recursive=True)
            notcars = glob.glob('non-vehicles_smallset/*/*.jpeg', recursive=True)
        else:
            cars = glob.glob('vehicles/*/*.png', recursive=True)
            notcars = glob.glob('non-vehicles/*/*.png', recursive=True)

        if self.sample_size > 0:
            cars = cars[0:self.sample_size]
            notcars = notcars[0:self.sample_size]

        logging.info('#vehicules: %d #nonVehicules: %d', len(cars),  len(notcars))
        t = time.time()
        car_features = self.imgs_features(cars)
        noncar_features = self.imgs_features(notcars)
        logging.info('%.2f Seconds to prepare features', time.time() - t)

        return car_features, noncar_features

    def train(self, car_features, notcar_features):
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        X_scaler = StandardScaler().fit(X)
        scaled_X = X_scaler.transform(X)

        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)

        logging.info('Feature vector length: %d', len(X_train[0]))

        model = LinearSVC(max_iter=10000, C=1)
        if self.grid:
            tuned_parameters = [{'C': [.1, 1, 10, 100]}]
            model = GridSearchCV(model, tuned_parameters, verbose=True, n_jobs=2)

        t = time.time()
        model.fit(X_train, y_train)

        logging.info('%.2f Seconds to fit model', time.time() - t)
        logging.info('Test Accuracy of SVC = %.2f', model.score(X_test, y_test))
        if self.grid:
            logging.info('best params found %s', model.best_params_)

        with open('svc.pickle', 'wb') as f:
            pickle.dump({'svc': model, 'scaler': X_scaler, 'cfg': self.cfg}, f)

    def train_svc(self):
        car_features, notcar_features = self.prepare_all_features()
        self.train(car_features, notcar_features)

