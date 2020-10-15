from utils import load_data
from config import config as cfg
from algorithms.FSFOA import FSFOA
from utils import get_CA, get_DR
import numpy as np

params = {
    'glass': {'LSC': 2, 'GSC': 4},
}


def main():
    X, Y = load_data(cfg.data)
    vector = FSFOA(X, Y, LSC=params[cfg.data]['LSC'], life_time=cfg.life_time,
          area_limit=cfg.area_limit, transfer_rate=cfg.transfer_rate, GSC=params[cfg.data]['GSC'])
    print(cfg.data, get_CA(vector, X, Y), get_DR(vector))

if __name__ == '__main__':
    main()
