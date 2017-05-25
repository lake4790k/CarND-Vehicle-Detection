import logging

from Config import Config
from Train import Train

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S', level=logging.INFO)
logging.info('Vehicule detection initializing')


def main():
    cfg = Config()
    train = Train(cfg)
    svc = train.train_svc()


if __name__ == '__main__':
    main()