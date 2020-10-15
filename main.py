from utils import load_data
from config import config as cfg
from algorithms.FSFOA import FSFOA

params = {
    'glass': {'LSC': 2, 'GSC': 4},
}


def main():
    X, Y = load_data(cfg.data)
    FSFOA(X, Y, LSC=params[cfg.data]['LSC'], life_time=cfg.life_time,
          area_limit=cfg.area_limit, transfer_rate=cfg.transfer_rate, GSC=params[cfg.data]['GSC'])


if __name__ == '__main__':
    main()
