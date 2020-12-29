import numpy as np

import utils.plot
import utils.process
from linear import conditions
from models import plot


def process_explicit_solution(
    borders=conditions.DEFAULT_BORDERS, N=20, J=1000
) -> plot.PlotData:
    h = (borders.right_border_x - borders.left_border_x) / (N - 1)
    tau = borders.right_border_t / (J - 1)
    assert tau <= h ** 2

    x, t = utils.process.get_x_and_t_arrays(borders=borders, N=N, J=J)

    u = np.zeros([len(t), len(x)])
    u[0] = conditions.u0(x)

    for j in range(J - 1):
        for i in range(1, N - 1):
            u[j + 1][i] = (tau / h ** 2) * (
                u[j][i - 1] - 2 * u[j][i] + u[j][i + 1]
            ) + u[j][i]

        u[j + 1][0] = conditions.u1(x[0])
        u[j + 1][N - 1] = conditions.u2(x[N - 1])

    xgrid, tgrid = np.meshgrid(x, t)
    return plot.PlotData(x=xgrid, t=tgrid, u=u)
