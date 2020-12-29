import numpy as np

from utils import process
from non_linear import conditions
from models import plot as plot_models


def k_half_j(u_j_i, u_j_i_next):
    return (conditions.k_u(u_j_i) + conditions.k_u(u_j_i_next)) / 2


def process_explicit_solution(
    borders=conditions.DEFAULT_BORDERS, N=20, J=1000
) -> plot_models.PlotData:
    h = (borders.right_border_x - borders.left_border_x) / (N - 1)
    tau = borders.right_border_t / (J - 1)
    assert tau <= h ** 2

    x, t = process.get_x_and_t_arrays(borders=borders, N=N, J=J)

    u = np.zeros([len(t), len(x)])
    u[0] = conditions.u0(x)

    for j in range(J - 1):
        for i in range(1, N - 1):
            u[j + 1][i] = (
                (tau / h ** 2) * k_half_j(u[j][i], u[j][i + 1]) * u[j][i + 1]
                + (tau / h ** 2) * k_half_j(u[j][i - 1], u[j][i]) * u[j][i - 1]
                - (tau / h ** 2) * (
                    k_half_j(u[j][i], u[j][i + 1]) * u[j][i]
                    + k_half_j(u[j][i - 1], u[j][i]) * u[j][i]
                )
                + u[j][i] + tau * conditions.f(u[j][i])
            )

    xgrid, tgrid = np.meshgrid(x, t)
    return plot_models.PlotData(x=xgrid, t=tgrid, u=u)
