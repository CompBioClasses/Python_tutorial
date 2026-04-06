'''
PRCC Sensitivity Analysis for an SIR Epidemic Model

This script performs Partial Rank Correlation Coefficient (PRCC) analysis
on an SIR model with temporary immunity. The output of interest is the
peak infected fraction, and PRCC tells us which parameters have the
strongest influence on that peak.

Concepts used:
    - Solving ODEs with scipy (solve_ivp)
    - Latin Hypercube Sampling
    - Rank-based sensitivity analysis (PRCC)
    - Functions, dictionaries, numpy arrays
    - try-except for robust model evaluation
    - Plotting with matplotlib

Usage: python sir_prcc.py

Author: Christopher Strickland
Math 682 - Computational Methods for Biomathematics
'''

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


###############################################################################
#                            Function definitions                             #
###############################################################################

def sir_rhs(t, y, params):
    '''Right-hand side of the SIR model with temporary immunity.

    The model is:
        dS/dt = -beta*S*I + mu*R
        dI/dt =  beta*S*I - gamma*I
        dR/dt =  gamma*I  - mu*R

    Parameters
    ----------
    t : float
        Current time (required by solve_ivp, even though the ODE is
        autonomous).
    y : array_like
        Current state [S, I, R].
    params : dict
        Must contain keys 'beta', 'gamma', 'mu'.

    Returns
    -------
    dydt : list
        Derivatives [dS/dt, dI/dt, dR/dt].
    '''

    S, I, R = y

    dSdt = -params['beta']*S*R + params['mu']*R
    dIdt = params['beta']*S*I - params['gaamma']*I
    dRdt = params['gamma']*I  - params['mu']*R

    return [dSdt, dIdt, dRdt]


def run_sir(params, S0, I0, R0, t_end):
    '''Solve the SIR model and return the peak infected fraction.

    Parameters
    ----------
    params : dict
        Model parameters with keys 'beta', 'gamma', 'mu'.
    S0, I0, R0 : float
        Initial conditions (population fractions).
    t_end : float
        End time for simulation.

    Returns
    -------
    peak_I : float
        Maximum value of I(t) over the simulation.
    '''

    y0 = [S0, I0, R0]
    t_span = (0, t_end)
    t_eval = np.linspace(0, t_end, 1000)

    sol = solve_ivp(sir_rhs, t_span, y0, t_eval=t_eval, args=(params,))

    # I is the second component: index 1
    peak_I = np.max(sol.y[2, :])

    return peak_I


def latin_hypercube_sample(n_samples, param_ranges):
    '''Generate Latin Hypercube Samples for the given parameter ranges.

    Latin Hypercube Sampling divides each parameter's range into n equal
    intervals and places exactly one sample in each interval, then shuffles
    so that the parameters are not correlated with each other. (Note: the
    result is not guaranteed to be a latin square, but this is not needed for 
    LHS-based sensitivity analysis.)

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    param_ranges : dict
        Dictionary mapping parameter names to (low, high) tuples.

    Returns
    -------
    samples : dict
        Dictionary mapping parameter names to arrays of sampled values.
    '''

    samples = {}

    for name, (low, high) in param_ranges.items():
        # Divide [0,1] into n_samples equal intervals
        intervals = np.linspace(0, 1, n_samples + 1)
        # Sample uniformly within each interval
        points = np.random.uniform(intervals[:-1], intervals[1:])
        # Shuffle to remove correlation between parameters
        np.random.shuffle(points)
        # Scale to the parameter range
        samples[name] = low + points * (high - low)

    return samples


def compute_prcc(param_samples, output):
    '''Compute Partial Rank Correlation Coefficients.

    PRCC measures the strength of the monotonic relationship between each
    input parameter and the output, after removing the linear effect of
    all other parameters. It operates on rank-transformed data.

    Algorithm:
        1. Rank-transform all input parameters and the output.
        2. For each parameter X_j:
           a. Regress ranked X_j on all other ranked parameters -> residual_x
           b. Regress ranked output on all other ranked parameters -> residual_y
           c. PRCC_j = Pearson correlation between residual_x and residual_y

    Parameters
    ----------
    param_samples : dict
        Dictionary mapping parameter names to arrays of sampled values.
    output : array_like
        Model output for each sample.

    Returns
    -------
    prcc_values : dict
        PRCC value for each parameter.
    '''

    param_names = list(param_samples.keys())
    n_samples = len(output)
    n_params = len(param_names)

    # Step 1: Rank-transform all variables using scipy's rankdata function
    ranked_params = np.zeros((n_samples, n_params))
    for j, name in enumerate(param_names):
        ranked_params[:, j] = rankdata(param_samples[name])
    ranked_output = rankdata(output)

    # Step 2: For each parameter, compute partial correlation
    prcc_values = {}
    for j, name in enumerate(param_names):
        # Indices of all OTHER parameters
        other_idx = [k for k in range(n_params) if k != j]
        C = ranked_params[:, other_idx]

        # Add intercept column for regression
        C_aug = np.column_stack([np.ones(n_samples), C])

        # Regress the parameter of interest on the other parameters
        coeffs_x, _, _, _ = np.linalg.lstsq(C_aug, ranked_params[:, j],
                                             rcond=None)
        residual_x = ranked_params[:, j] - C_aug @ coeffs_x

        # Regress the output on the other parameters
        coeffs_y, _, _, _ = np.linalg.lstsq(C_aug, ranked_output, rcond=None)
        residual_y = ranked_output - C_aug @ coeffs_y

        # PRCC = correlation between the two residuals
        prcc = np.corrcoef(residual_x, residual_y)[0, 1]
        prcc_values[name] = prcc

    return prcc_values


def plot_prcc(prcc_values):
    '''Create a bar chart of PRCC values.

    Parameters
    ----------
    prcc_values : dict
        PRCC value for each parameter.
    '''

    names = list(prcc_values.keys())
    values = list(prcc_values.values())

    # Color bars by sign: blue for positive, red for negative
    colors = ['steelblue' if v >= 0 else 'indianred' for v in values]

    plt.figure(figsize=(8, 5))
    plt.bar(names, values, color=colors)
    plt.axhline(y=0, color='black', linewidth=0.5)
    plt.ylabel('PRCC')
    plt.title('PRCC Sensitivity Analysis: SIR Model\n'
              '(Output: Peak Infected Fraction)')
    plt.ylim(-1, 1)
    plt.tight_layout()


###############################################################################
#                               Main script                                   #
###############################################################################

if __name__ == '__main__':

    # Seed the random number generator for reproducibility
    np.random.seed(42)

    # --- Parameter ranges for the sensitivity analysis ---
    # Each parameter is sampled uniformly over its range via LHS.
    param_ranges = {
        'beta':  (0.3, 3.0),    # transmission rate
        'gamma': (0.05, 0.5),   # recovery rate
        'mu':    (0.005, 0.1)   # immunity loss rate
    }

    # --- Fixed initial conditions (population fractions) ---
    I0 = 1e-3
    R0 = 0.0
    S0 = 1 - I0 - R0
    t_end = 200

    # --- Generate Latin Hypercube Samples ---
    n_samples = 500
    print(f"Generating {n_samples} Latin Hypercube samples...")
    samples = latin_hypercube_sample(n_samples, param_ranges)

    # --- Run the model for each parameter set ---
    print("Running SIR model for each parameter set...")
    peak_infections = np.zeros(n_samples)

    for i in range(n_samples):
        params = {name: samples[name][i] for name in param_ranges}
        try:
            peak_infections[i] = run_sir(params, S0, I0, R0, t_end)
        except:
            peak_infections[i] = np.nan

    # --- Check for failed runs ---
    n_failed = np.sum(np.isnan(peak_infections))
    if n_failed > 0:
        print(f"Warning: {n_failed} out of {n_samples} model runs failed.")
    else:
        print("All model runs completed successfully.")

    # --- Compute PRCC (exclude any failed runs) ---
    valid = ~np.isnan(peak_infections)
    valid_samples = {name: samples[name][valid] for name in param_ranges}
    valid_output = peak_infections[valid]

    print("Computing PRCC values...")
    prcc = compute_prcc(valid_samples, valid_output)

    # --- Print results ---
    print("\n--- PRCC Results ---")
    print(f"{'Parameter':<10} {'PRCC':>8}")
    print("-" * 20)
    for name, value in prcc.items():
        print(f"{name:<10} {value:>8.4f}")

    # --- Plot and display ---
    plot_prcc(prcc)
    plt.show()
