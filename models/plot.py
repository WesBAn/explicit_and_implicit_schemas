import dataclasses

import numpy as np


@dataclasses.dataclass
class PlotData:
    x: np.array
    t: np.array
    u: np.array
