from . import utils
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import logging

logging.basicConfig(
    filename=os.path.join(ROOT_DIR, "output/method.log"),
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
