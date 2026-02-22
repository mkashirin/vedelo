import os
from pathlib import Path

import torch as pt

from .common import *


_DEPTH: int = 1
ROOT: Path = Path(__file__).parents[_DEPTH].resolve()
PYTORCH: str = os.path.dirname(pt.__file__)
