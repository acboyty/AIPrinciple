from utils import load_data
from config import config as cfg
from algorithms.FSFOA import FSFOA
from utils import get_CA, get_DR
import numpy as np

params = {
    'heart': {'LSC': 3, 'GSC': 6},
    'cleveland': {'LSC': 3, 'GSC': 6},
    'dermatology': {'LSC': 7, 'GSC': 15},
    'ionosphere': {'LSC': 7, 'GSC': 15},
    'wine': {'LSC': 3, 'GSC': 6},
    'vehicle': {'LSC': 4, 'GSC': 9},
    'sonar': {'LSC': 12, 'GSC': 30},
    'glass': {'LSC': 2, 'GSC': 4},
    'segmentation': {'LSC': 4, 'GSC': 9},
    'hepatitis': {'LSC': 4, 'GSC': 10},
}


def main():
    X, Y = load_data(cfg.data)
    vector, training_info = FSFOA(X, Y, LSC=params[cfg.data]['LSC'], life_time=cfg.life_time,
                   area_limit=cfg.area_limit, transfer_rate=cfg.transfer_rate, GSC=params[cfg.data]['GSC'])
    np.save(f'./training_info/{cfg.data}.npy', training_info)
    print(cfg.data, get_CA(vector, X, Y), get_DR(vector))


if __name__ == '__main__':
    main()
