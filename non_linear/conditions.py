import numpy as np

from models import linear_borders

# u_t = (2u*u_x)_x + u^2
Lt = 4 * np.pi
X_NON_ZERO_LEFT = -Lt / 2
X_NON_ZERO_RIGHT = Lt / 2
T = 0.29

X0 = X_NON_ZERO_LEFT - 2
XN = X_NON_ZERO_RIGHT + 2
T0 = 0.3


DEFAULT_BORDERS = linear_borders.Borders(
    left_border_x=X0, right_border_x=XN, right_border_t=T,
)


def k_u(u):
    return 2 * u


def f(u):
    return u ** 2


def u0(x):
    return np.array(
        [
            4 / 3 * np.power(np.cos(x[i] / 4), 2) / T0
            if _check_x_in_Lt_range(x[i])
            else 0
            for i in range(len(x))
        ]
    )


def u1(t):
    return 0


def u2(t):
    return 0


def u(x, t: np.array):
    assert all(t < T0)
    u = np.zeros([len(t), len(x)])
    for j in range(len(t)):
        for i in range(len(x)):
            if _check_x_in_Lt_range(x[i]):
                u[j][i] = 4 / 3 * np.power(np.cos(x[i] / 4), 2) / (T0 - t[j])
    return u


def _check_x_in_Lt_range(x: float):
    return X_NON_ZERO_LEFT < x < X_NON_ZERO_RIGHT
