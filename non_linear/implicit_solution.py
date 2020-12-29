import numpy as np

import utils.plot
import utils.process
from non_linear import conditions
from models import plot


def k_half_j(u_j_i, u_j_i_next):
    return (conditions.k_u(u_j_i) + conditions.k_u(u_j_i_next)) / 2


def A_i(tau, h, i, u_j):
    return tau / h ** 2 * k_half_j(u_j[i - 1], u_j[i])


def B_i(tau, h, i, u_j):
    return tau / h ** 2 * k_half_j(u_j[i], u_j[i + 1])


def C_i(tau, h, i, u_j):
    return (
        tau / h ** 2 * (k_half_j(u_j[i - 1], u_j[i]) + k_half_j(u_j[i], u_j[i + 1])) + 1
    )


def F_i(tau, i, u_prev):
    return u_prev[i] + tau * conditions.f(u_prev[i])


def get_alphas(tau, h, N, u_prev):
    alphas = np.zeros(N - 1)
    alphas[0] = conditions.u1(alphas[0])
    for i in range(1, N - 1):
        alphas[i] = B_i(tau, h, i, u_prev) / (
            C_i(tau, h, i, u_prev) - alphas[i - 1] * A_i(tau, h, i, u_prev)
        )
    return alphas


def get_betas(tau, h, N, alphas, u_prev: np.array):
    betas = np.zeros(N)
    betas[0] = conditions.u1(u_prev[0])
    for i in range(1, N - 1):
        betas[i] = (A_i(tau, h, i, u_prev) * betas[i - 1] + F_i(tau, i, u_prev)) / (
            C_i(tau, h, i, u_prev) - alphas[i - 1] * A_i(tau, h, i, u_prev)
        )
    return betas


def process_implicit_solution(
    borders=conditions.DEFAULT_BORDERS, N=10, J=100
) -> plot.PlotData:
    h = (borders.right_border_x - borders.left_border_x) / (N - 1)
    tau = borders.right_border_t / (J - 1)

    x, t = utils.process.get_x_and_t_arrays(borders=borders, N=N, J=J)
    u = np.zeros([len(t), len(x)])
    u[0] = conditions.u0(x)

    completed = 0
    percent_completed = 0
    for j in range(J - 1):
        alphas = get_alphas(tau=tau, h=h, N=N, u_prev=u[j])
        betas = get_betas(tau=tau, h=h, N=N, alphas=alphas, u_prev=u[j])
        u[j + 1][N - 1] = 0
        for i in range(N - 2, -1, -1):
            u[j + 1][i] = alphas[i] * u[j + 1][i + 1] + betas[i]
        # print(f't={t[j]}, u={u[j]}')
        completed += 1
        if int(completed / J * 100) > percent_completed:
            percent_completed = int(completed / J * 100)
            print(
                f'{percent_completed}%',
                end=' .. ' if percent_completed % 10 != 0 else '\n',
            )

    xgrid, tgrid = np.meshgrid(x, t)
    print('100%')
    return plot.PlotData(x=xgrid, t=tgrid, u=u)
