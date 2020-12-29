from linear import analytical_solution as lin_analytical_solution
from linear import conditions as lin_conditions
from linear import explicit_solution as lin_explicit_solution
from linear import implicit_solution as lin_implicit_solution
from non_linear import analytical_solution as non_lin_analytical_solution
from non_linear import conditions as non_lin_conditions
from non_linear import implicit_solution as non_lin_implicit_solution
from non_linear import explicit_solution as non_lin_explicit_solution
from utils import process


def process_linear():
    lin_explicit_data = process.process(
        lin_explicit_solution.process_explicit_solution,
        J=1000,
        N=20,
        borders=lin_conditions.DEFAULT_BORDERS,
        title='Linear Explicit',
        eps=0.001,
    )

    process.process_comparison(
        lin_analytical_solution.process_analytical_solution,
        borders=lin_conditions.DEFAULT_BORDERS,
        N=lin_explicit_data.N,
        J=lin_explicit_data.J,
        approx_data=lin_explicit_data,
        title_analyt='Linear Analytical',
        title_err='Linear Explicit Error',
    )

    lin_implicit_data = process.process(
        lin_implicit_solution.process_implicit_solution,
        J=125,
        N=20,
        borders=lin_conditions.DEFAULT_BORDERS,
        title='Linear Implicit',
        eps=0.001,
    )

    process.process_comparison(
        lin_analytical_solution.process_analytical_solution,
        borders=lin_conditions.DEFAULT_BORDERS,
        N=lin_implicit_data.N,
        J=lin_implicit_data.J,
        approx_data=lin_implicit_data,
        title_analyt='Linear Analytical',
        title_err='Linear Implicit Error',
    )


def process_non_linear():
    non_lin_implicit_data = process.process(
        non_lin_implicit_solution.process_implicit_solution,
        J=1000,
        N=200,
        borders=non_lin_conditions.DEFAULT_BORDERS,
        title='Non-linear Implicit',
        eps=0.2,
    )

    process.process_comparison(
        non_lin_analytical_solution.process_analytical_solution,
        borders=non_lin_conditions.DEFAULT_BORDERS,
        N=non_lin_implicit_data.N,
        J=non_lin_implicit_data.J,
        approx_data=non_lin_implicit_data,
        title_analyt='Non-linear Analytical',
        title_err='Non-linear Implicit Error',
    )

    non_lin_explicit_data = process.process(
        non_lin_explicit_solution.process_explicit_solution,
        J=1000,
        N=20,
        borders=non_lin_conditions.DEFAULT_BORDERS,
        title='Non-linear Explicit',
        eps=0.5,
    )

    process.process_comparison(
        non_lin_analytical_solution.process_analytical_solution,
        borders=non_lin_conditions.DEFAULT_BORDERS,
        N=non_lin_explicit_data.N,
        J=non_lin_explicit_data.J,
        approx_data=non_lin_explicit_data,
        title_analyt='Non-linear Analytical',
        title_err='Non-linear Explicit Error',
    )


def main():
    process_linear()
    process_non_linear()


if __name__ == "__main__":
    main()
