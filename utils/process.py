import typing

import numpy as np

from models import linear_borders
from models import process_result
from models import plot as plot_models
from utils import plot as plot_utils


def process(
    process_func, *, J, N, borders: linear_borders, title: str, eps: float
) -> process_result.ProcessResponse:
    print(f'Start processing {title}')
    norm = 1.0
    u_next = process_func(borders=borders, J=J, N=N)

    tau = borders.right_border_t / (J - 1)
    while norm > eps:
        oldJ = J
        J = 2 * J
        print('Current J: ', J)
        tau = borders.right_border_t / (J - 1)
        u_tau = process_func(borders=borders, J=J, N=N)

        norm = 0.0
        for j in range(oldJ // 2, oldJ):
            err = np.max(
                [np.fabs(u_next.u[j][i] - u_tau.u[2 * j][i]) for i in range(N)]
            )
            norm = np.fmax(err, norm)

        print('Norm = ', norm, end='\n\n')
        u_next = u_tau

    print(f'Result norm = {norm}')
    print(f'Tau = {tau}')
    print(f'J = {J}')
    print(f'Accuracy {eps}')

    plot_utils.plot_u_x_t(
        plot_data=u_next, borders=borders, title=title, value_title='u'
    )
    print(f'End processing {title}')
    return process_result.ProcessResponse(
        N=N, J=J, xgrid=u_next.x, tgrid=u_next.t, ugrid=u_next.u,
    )


def process_comparison(
    process_analyt_func, *, borders, N, J, approx_data, title_analyt, title_err
):
    analytical_data = process_analyt_func(borders=borders, N=N, J=J)
    plot_utils.plot_u_x_t(
        plot_data=analytical_data, borders=borders, title=title_analyt, value_title='u'
    )
    error_data = plot_models.PlotData(
        x=approx_data.xgrid,
        t=approx_data.tgrid,
        u=analytical_data.u - approx_data.ugrid,
    )
    plot_utils.plot_u_x_t(
        plot_data=error_data, borders=borders, title=title_err, value_title='error'
    )


def get_x_and_t_arrays(
    borders: linear_borders.Borders, *, N: int, J: int
) -> typing.Tuple[np.array, np.array]:
    h_x = (borders.right_border_x - borders.left_border_x) / (N - 1)
    x = np.array([borders.left_border_x + i * h_x for i in range(N)])
    assert len(x) == N, len(x)

    h_t = borders.right_border_t / (J - 1)
    t = np.array([j * h_t for j in range(J)])
    assert len(t) == J, len(t)

    return x, t
