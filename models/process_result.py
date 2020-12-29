import numpy as np
import dataclasses


@dataclasses.dataclass
class ProcessResponse:
    N: int
    J: int
    xgrid: np.array
    tgrid: np.array
    ugrid: np.array
