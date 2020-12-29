import numpy as np

import utils.plot
import utils.process
from non_linear import conditions
from models import plot


def process_analytical_solution(
    *, borders=conditions.DEFAULT_BORDERS, N=10, J=100
) -> plot.PlotData:
    x, t = utils.process.get_x_and_t_arrays(borders=borders, N=N, J=J)
    ugrid = conditions.u(x, t)
    xgrid, tgrid = np.meshgrid(x, t)

    return plot.PlotData(x=xgrid, t=tgrid, u=ugrid)


if __name__ == '__main__':
    process_analytical_solution()
