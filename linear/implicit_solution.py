import numpy as np

import utils.plot
import utils.process
from linear import conditions
from models import plot


def A_i(tau, h):
    return tau / h ** 2


def B_i(tau, h):
    return tau / h ** 2


def C_i(tau, h):
    return 2 * tau / h ** 2 + 1


def get_alphas(tau, h, N):
    alphas = np.zeros(N - 1)
    alphas[0] = conditions.u1(alphas[0])
    for i in range(1, N - 1):
        alphas[i] = B_i(tau, h) / (C_i(tau, h) - alphas[i - 1] * A_i(tau, h))
    return alphas


def get_betas(tau, h, N, alphas, f_arr: np.array):
    betas = np.zeros(N)
    betas[0] = conditions.u1(f_arr[0])
    for i in range(1, N):
        betas[i] = (A_i(tau, h) * betas[i - 1] + f_arr[i]) / (
            C_i(tau, h) - alphas[i - 1] * A_i(tau, h)
        )
    return betas


def process_implicit_solution(
    borders=conditions.DEFAULT_BORDERS, N=100, J=100
) -> plot.PlotData:
    h = (borders.right_border_x - borders.left_border_x) / (N - 1)
    tau = borders.right_border_t / (J - 1)

    x, t = utils.process.get_x_and_t_arrays(borders=borders, N=N, J=J)

    u = np.zeros([len(t), len(x)])
    u[0] = conditions.u0(x)

    for j in range(J - 1):
        alphas = get_alphas(tau=tau, h=h, N=N)
        betas = get_betas(tau=tau, h=h, N=N, alphas=alphas, f_arr=u[j])
        u[j + 1][N - 1] = 0
        for i in range(N - 2, -1, -1):
            u[j + 1][i] = alphas[i] * u[j + 1][i + 1] + betas[i]

    xgrid, tgrid = np.meshgrid(x, t)
    return plot.PlotData(x=xgrid, t=tgrid, u=u)
