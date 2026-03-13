'''
Demonstration of Python scripting (as opposed to using Jupyter notebooks).

This script shows how to write a Python .py file that can be run from a terminal
with the command: python scripting_new.py

Key concepts demonstrated:
    - Comments (single-line and multi-line)
    - Imports and script structure
    - Plotting with matplotlib (figures pop up in separate windows)
    - The "with" keyword (context managers)
    - Solving an ODE with scipy
    - Organizing code into functions (call stack)
    - The if __name__ == "__main__" idiom

Author: Christopher Strickland
'''

# Single-line comments start with #. Use them liberally!
# "We write code to be read by humans, not machines."
#
# The triple-quoted string at the top of this file is a multi-line comment,
# also called a "docstring." Python uses these for documentation:
#   - At the top of a file: describes what the file does
#   - At the top of a function: describes what the function does
# Docstrings are special because Python stores them and you can access them
# with help(). For example, try: help(solve_logistic) in a Python prompt
# after importing this file.
#
# You can also use triple quotes for multi-line comments in the middle of
# code, but by convention those are reserved for docstrings. Use # for
# regular comments.

# Place all imports at the top of your script. Import only what you need.
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


###############################################################################
#                            Function definitions                             #
###############################################################################

def logistic_eqn(t, x, params):
    '''Right-hand side of the logistic ODE: dx/dt = r*x*(1 - x/K).

    Parameters
    ----------
    t : float
        Current time (required by solve_ivp, even if the ODE is autonomous).
    x : array_like
        Current state.
    params : dict
        Must contain keys 'r' (growth rate) and 'K' (carrying capacity).

    Returns
    -------
    dxdt : array_like
    '''

    dxdt = params['r']*x*(1 - x/params['K'])

    return dxdt


def solve_logistic(x0, tstart, tstop, params):
    '''Solve the logistic equation and return the solution object.

    This function calls logistic_eqn, so when debugging you will see both
    functions in the call stack (try setting a breakpoint inside logistic_eqn).

    Parameters
    ----------
    x0 : float
        Initial population.
    tstart : float
        Start time.
    tstop : float
        End time.
    params : dict
        Parameters for logistic_eqn.

    Returns
    -------
    solution : OdeSolution
        Object returned by solve_ivp with attributes .t and .y
    '''

    y0 = np.array([x0])  # solve_ivp requires an array for the initial condition

    # The default solver is RK45 — an explicit, variable-step Runge-Kutta
    # method of order 4(5), also known as Dormand-Prince (basically ODE45).
    # args passes extra arguments to logistic_eqn. It must be a tuple.
    solution = solve_ivp(logistic_eqn, t_span=[tstart, tstop], y0=y0,
                         args=(params,))

    return solution


def plot_solution(sol):
    '''Plot the ODE solution in a new figure window.

    Parameters
    ----------
    sol : OdeSolution
        Solution object from solve_ivp.
    '''

    time = sol.t
    x = sol.y[0, :]  # one equation, so .y has shape (1, n_timepoints)

    plt.figure()
    plt.plot(time, x, label='logistic eqn')
    plt.xlabel('time')
    plt.ylabel('x(t)')
    plt.legend()
    plt.tight_layout()
    # plt.show() is NOT called here — we'll show all figures at the end.
    # This lets us build up multiple figures before pausing the script.


###############################################################################
#                               Main script                                   #
###############################################################################

# This if-statement is standard boilerplate. It says:
#   "Only run the code below when this file is executed directly."
# If someone imports this file as a module (e.g. "import scripting_new"),
# the functions above will be available but the code below will NOT run.
# This is how you make a file that is both a reusable module and a standalone
# script.

if __name__ == "__main__":

    # --- Part 1: A simple plot to demonstrate plt.show() ---
    # When you run a .py script, plots do not appear automatically the way
    # they do in Jupyter notebooks. You must call plt.show() to display them.
    # Each figure pops up in its own window, and the script PAUSES until you
    # close the window (or all open windows, depending on the backend).

    x_vals = np.linspace(-10, 10, 200)
    y_vals = x_vals**2 - 10

    plt.figure()
    plt.plot(x_vals, y_vals)
    plt.title('Parabola')
    plt.xlabel('x')
    plt.ylabel('y')

    # Calling plt.show() here will pause the script and display the figure.
    # Close the plot window to continue.
    plt.show()

    print("You closed the first plot. Continuing...")

    # --- Part 2: The "with" keyword (context managers) ---
    # "with" creates a temporary context that is automatically cleaned up
    # when the block ends. Common uses:
    #   - Opening files: the file is automatically closed when the block ends
    #   - Temporary settings: restored when the block ends
    #
    # File example (the most important use case):
    with open('demo_with.txt', 'w') as f:
        f.write('This file was created using a with statement.\n')
        f.write('The file is automatically closed when the with block ends.\n')
    # f is now closed — no need to call f.close() manually.
    # This is safer because the file gets closed even if an error occurs
    # inside the with block.

    # Read it back to verify:
    with open('demo_with.txt', 'r') as f:
        contents = f.read()
    print(contents)

    # Fun plotting example: plt.xkcd() is a context manager that temporarily
    # switches to xkcd-style (hand-drawn) plots.
    with plt.xkcd():
        plt.figure(figsize=(6, 4))
        plt.plot(x_vals, np.sin(x_vals), label='sin(x)')
        plt.title('A hand-drawn plot')
        plt.legend()
        plt.tight_layout()
    # Outside the with block, plotting style returns to normal.

    # We haven't called plt.show() yet, so the xkcd figure is waiting.

    # --- Part 3: Solve the logistic ODE ---
    # Set up parameters using a dictionary
    params = {}
    params['r'] = 1       # intrinsic growth rate
    params['K'] = 1000    # carrying capacity

    x0 = 1       # initial population
    tstart = 0   # start time
    tstop = 15   # end time

    # solve_logistic calls logistic_eqn internally.
    # If you set a breakpoint inside logistic_eqn (e.g. with pdb), you will
    # see both solve_logistic and logistic_eqn in the call stack.
    sol = solve_logistic(x0, tstart, tstop, params)

    # Plot the result (this creates a figure but does not show it yet)
    plot_solution(sol)

    # Now show ALL remaining figures at once. Close them to end the script.
    plt.show()
