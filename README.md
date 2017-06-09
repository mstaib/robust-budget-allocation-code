# Robust Budget Allocation via Continuous Submodular Functions
This repository contains the supporting code for the paper:

[Staib, Matthew and Jegelka, Stefanie. Robust Budget Allocation via Continuous Submodular Functions. In _Proceedings of the 34th International Conference on Machine Learning_, 2017.](https://arxiv.org/abs/1702.08791)

## Dependencies
* [TFOCS](https://github.com/cvxr/TFOCS)
* [MOSEK](https://www.mosek.com/) (for the first-order comparison experiments)
* Yahoo! Webscope dataset ydata-ysm-advertiser-bids-v1.0 (for some experiments)

## Getting started
1. First run `addpaths` to add subdirectories to the path
2. Run `mexMake` to build mex code
3. If you plan to use the Yahoo! data, update line 75 of `experiments/get_yahoo_problem_func.m` to point to the location of this dataset on your system. (the default is `../Webscope_A1/ydata-ysm-advertiser-bids-v1_0.txt`)

At this point you can run the experiments from the paper, or try your own by passing different parameter combinations into `experiments/run_all_params`. Plotting utilities are available for some specific experiments already:
* `experiments/rho_uniqueness_experiment.m` produces Figure 1 from the paper, which tests the uniqueness of values in the vector rho which we threshold in order to recover solutions to the constrained SFM problem.
* Assuming you have already run the budget allocation code on synthetic experiments, `experiments/synthetic_influence_comparison_plots.m` will produce Figure 2 from the paper.
* `experiments/plot_yahoo_convergence.m` produces Figure 3 from the paper (again assuming the budget allocation code was run already).
* `experiments/gradient_descent_yahoo_compare.m` produces Figure 4 from the paper (again assuming the budget allocation code was run already) (requires MOSEK).
