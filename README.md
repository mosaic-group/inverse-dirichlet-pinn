# Inverse-Dirichlet Weighting Enables Reliable Training of Physics Informed Neural Networks

This repository contains the supplementary code to the manuscript [here](https://arxiv.org/abs/2107.00940). The codes should be sufficient to reproduce all the results presented.

**Dependencies:** `numpy`, `scipy`, `torch`, `matplotlib`, `seaborn`, `pandas`

## Contents

### Poisson

* `PoissonExample`: fully runnable code example for all weighting strategies on the Poisson equation for different modes.
* `poisson_adam.py`, `poisson_rms.py`, `poisson_ada.py`: codes to run inference of poisson solutions for different modes and optimizers as reported in Appendix D.
* `EvaluationAdam`, `EvaluationRMS`, `EvaluationAdaGrad`: evaluate results from `poisson_*.py` to generate plots from paper

### Sobolev

* `sobolev.py`: run inference on sobolev training example.

### Active Turbulence

The training data is publicly available [here](https://cloud.mpi-cbg.de/index.php/s/sFPQ49WQcbLgEAZ).

* `solve_vort_torus.py`, `solve_vort_square.py`: Infers solution of active turbulence problem in annular and squared domain.
* `eval_solve_vort_torus.py`, `eval_solve_vort_square.py`: Evaluation scripts for solution code.
* `eval_solve_vort_square_gradient.py`: Retrieve backpropagated gradients for solution in squared domain.
* `solve_vort_square_convergence_rand.py`: Perform convergence study for forward solution.
* `inference_pressure_catastrophic.py`: Inference of model parameters and effictive pressure with catastrophic interference.
* `timing_forward.py`, `timing_inverse.py`: Time comparison for different methods in Appendix C.
* `activation_reconstruction.py`, `activation_evaluation.py`: Reconstruction and evaluation of the data under different activation functions as in Appendix C.
* Notebooks produce plots for errors in forward and inverse modeling problems.
