from Common import *
from FindCars import *
import collections
from scipy.ndimage.measurements import label

logging.info('%s initializing', __name__)


class StableCars:

    def __init__(self, find: FindCars, alpha=0.99, last_n=10, threshold=10):
        self.find = find
        self.history = collections.deque(maxlen=last_n)
        self.alpha = alpha
        self.threshold = threshold
        self.heat = np.zeros([])

    def stablize(self, img):
        bboxes = self.find.find_cars(img)

        self.history.append(bboxes)
        self.heat = np.zeros_like(img[:, :, 0]).astype(np.float)

        decay = 1
        for bboxes in reversed(self.history):
            for bbox in bboxes:
                self.heat[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]] += decay

            decay *= self.alpha

        np.clip(self.heat, 0, 255)
        self.heat[self.heat < self.threshold] = 0

        labels = label(self.heat)
        for car_number in range(1, labels[1] + 1):
            nonzero = (labels[0] == car_number).nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            cv2.rectangle(img, bbox[0], bbox[1], (0, 255, 0), 6)

        return img

    def draw_heat(self, img):
        self.stablize(img)

        return self.heat
