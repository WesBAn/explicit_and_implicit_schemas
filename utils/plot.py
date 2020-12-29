import matplotlib.pyplot as plt

from models import linear_borders
from models import plot


def plot_u_x_t(
    *,
    plot_data: plot.PlotData,
    borders: linear_borders.Borders,
    title: str,
    value_title: str
) -> None:
    xgrid = plot_data.x
    tgrid = plot_data.t
    ugrid = plot_data.u
    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    plt.title(title)
    axes.set_xlabel('x')
    axes.set_xlim(borders.left_border_x, borders.right_border_x)
    axes.set_ylabel('t')
    axes.set_ylim(0, borders.right_border_t)
    axes.set_zlabel(value_title)
    axes.plot_surface(xgrid, tgrid, ugrid, cmap='plasma')
    plt.show()
