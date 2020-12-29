import numpy as np

from models import linear_borders

X0 = 0
XN = np.pi
T = 0.5

DEFAULT_BORDERS = linear_borders.Borders(
    left_border_x=X0, right_border_x=XN, right_border_t=T,
)


def u0(x):
    return 3 * np.sin(2 * x)


def u1(x):
    return 0


def u2(x):
    return 0


def u(x, t):
    return np.array([3 * np.exp(-4 * t[j]) * np.sin(2 * x) for j in range(len(t))])
