import logging

from Config import Config
from Train import Train

logging.info('%s initializing', __name__)


def main():
    cfg = Config()
    train = Train(cfg)
    svc = train.train_svc()


if __name__ == '__main__':
    main()