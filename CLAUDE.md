# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational Python tutorial for University of Tennessee Math 682. Teaches Python fundamentals through mathematical modeling, ODE systems, and epidemiological models (SIR, SVIR, opioid SPAR model). Author: Christopher Strickland.

This codebase is being reworked to support a new plan and calendar for the class. 
In the new format, Jupyter notebooks will be posted to the course website as worksheet-style 
lessons that the instructor then goes through. These notebooks include tutorial-style 
information but also exercises for the students to work through in-class. The new notebooks 
are appended with a number that denotes the order in which they will appear in the course. 
Each class is expected to be 1 hour and 15 minutes long, and the class will go through at most 
one Jupyter notebook per class. Some classes will not use a Jupyter notebook - the instructor is 
also writing lesson notes elsewhere and has other tasks that are not in this repository.

Files that are not prepended with a number are either old (used for reference when creating the new notebooks), 
supporting files for use by the instructor, or data.

## Environment & Dependencies

- Python 3.5+ (Conda-managed, configured in .vscode/settings.json)
- Core: numpy, scipy, matplotlib
- Advanced: SALib (sensitivity analysis, install via `conda install -c conda-forge salib`), pandas, multiprocessing (stdlib)

## Running Code

- **Jupyter notebooks** (numbered 0-5 plus supplementary): Primary teaching medium, run cells sequentially
- **Standalone scripts**: `python scriptname.py` (e.g., `python SIR.py`, `python aparse.py 3 -c blue`)
- **Profiling**: `python -m cProfile -o sl.prof slow_prog.py` then `snakeviz sl.prof`
- **Sensitivity analysis**: `python Run_Sobol.py -N 1000 -n 4 -o analysis`
- **Parallel runs**: `python SIR_parallel.py` (uses multiprocessing.Pool)

## Architecture

- **Notebooks** (`*.ipynb`): Progressive curriculum from data structures → functions → loops → conditionals → branching processes → ODE systems → data fitting
- **ODE model scripts** (`SIR.py`, `SVIR.py`, `opioid_model.py`): All use `scipy.integrate.solve_ivp` (not the legacy `ode` interface) with dictionary-based parameter passing (`params = {}`)
- **OOP examples** (`oop_cal.py`, `oop_twodice.py`): Class definitions with inheritance, `@property`/`@setter` decorators
- **Sensitivity analysis** (`Run_Sobol.py`, `opioid_sensitivity/`): SALib-based Sobol analysis with argparse CLI

## Key Conventions

- ODE model functions accept `(t, y, params)` where `params` is a dict
- Parameter sweeps use `multiprocessing.Pool` for parallelization (see `SIR_parallel.py`)
- Data files live in `01HIVseries/` (HIV time series), `epi_data/` (mortality/vaccination data), and root-level `.npy`/`.csv` files
