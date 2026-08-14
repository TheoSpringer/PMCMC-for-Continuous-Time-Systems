Summary
This repository accompanies the paper "PMCMC for Continuous-Time Systems" and provides an implementation of the experiments described there. This README focuses on the glucose example described in the paper and contains usage notes, background, and guidance for reproducing the example results. The implemented simulation script is provided at examples/1_Simulation.jl.

Contents

Project overview
Requirements and setup
Running the glucose example
What the glucose example demonstrates
Implementation notes (files & structure)
Typical outputs and interpretation
Reproducibility tips and troubleshooting
Citation
Project overview
This project implements particle MCMC (PMCMC) techniques for inference and state estimation in continuous-time dynamical systems. The glucose example demonstrates the method on a glucose–insulin style model (see the paper for derivation and model equations). The example shows how to simulate data under the model, apply the PMCMC sampler, and inspect inference results (posterior parameter estimates, latent-state trajectories, and predictive uncertainty).

Requirements and setup

Julia (recommended version: the one used for development; if unsure, use Julia 1.8+)
A working internet connection to resolve Julia packages the first time
Basic command-line familiarity
Quick setup (from project root)

Open a terminal in the repository root.
Instantiate the project environment (this installs the packages used by the project):
julia --project=. -e 'using Pkg; Pkg.instantiate()'
If the example scripts require plotting back-ends or JLD2, the package instantiation will install them.
Running the glucose example
Location of the example simulation script:

examples/1_Simulation.jl
Note: If your local copy places the script somewhere else, substitute the correct path. The repository examples directory should contain the implemented example script.

Typical run steps

Ensure packages are installed:
julia --project=. -e 'using Pkg; Pkg.instantiate()'
Run the simulation & example:
julia --project=. examples/1_Simulation.jl
What to expect

The example script will simulate glucose measurements under the model described in the paper.
It will run the PMCMC sampler to estimate posterior distributions over parameters and latent states.
The script typically produces:
Saved sampler traces (e.g., JLD2 or .csv files with samples)
Diagnostic output (acceptance rates, effective sample sizes, runtime)
Plots: posterior histograms, trace plots, predictive trajectories vs simulated observations
What the glucose example demonstrates (conceptual)

Simulating data from a continuous-time stochastic dynamical model for glucose dynamics (process model + measurement model).
Using a particle filter to estimate likelihoods of parameter proposals given discrete and noisy observations.
Wrapping the particle filter inside an MCMC scheme (e.g., particle marginal Metropolis–Hastings, PMMH) to sample from the posterior over parameters and latent states.
Handling missing/irregular observation times typical in physiological data.
Interpreting posterior uncertainty for model parameters and the inferred latent glucose trajectory.
Implementation notes (what to look for in the code)

examples/1_Simulation.jl: top-level script that runs the glucose simulation + inference pipeline. It will usually:
Set random seeds / experiment settings
Define or load model parameters (true values used for simulation)
Simulate the latent continuous-time process and sample observations
Configure the particle filter (number of particles, resampling threshold)
Configure the PMCMC sampler (number of iterations, burn-in, proposal covariance)
Run PMCMC and write outputs (samples, diagnostics, and plots)
packages/ScenarioPMCMC/: likely contains the PMCMC and particle filtering implementation used by the examples (particle filter, proposal kernels, helper functions).
examples/ (other example scripts): useful for reference on how other models are set up and run.
Key parameters and their roles (paper-backed)

Number of particles (Nparticles): higher N reduces particle approximation variance but increases cost.
PMCMC chain length (Niter): should be large enough to achieve ESS and convergence.
Proposal covariance (or scaling): critical for mixing; tuning via pilot runs/sampler_tuning.jl is recommended.
Measurement noise / observation model parameters: directly affect identifiability of latent states.
Interpreting outputs

Trace plots: check for mixing; aim for visually mixing traces and no long trends.
Posterior histograms & credible intervals: compare posterior means/medians with ground-truth simulation parameter values.
Latent-state plots: show posterior predictive bands (e.g., 95% credible intervals) and the true simulated latent trajectory; these illustrate how well the method reconstructs the hidden glucose process from noisy observations.
Diagnostics: effective sample size (ESS), acceptance rate for PMMH; low ESS or very low acceptance suggests increasing Nparticles or tuning the proposal.
Reproducibility tips

Fix random seeds in the script if you want exact reproducibility for the simulated data. However, PMCMC samplers are stochastic — exact replicability of sample draws requires the same RNG streams and identical package versions.
Use the Project.toml / Manifest.toml in the repository root to pin package versions. If those files are present, Pkg.instantiate() should restore the environment used for development.
For long-running PMCMC jobs, save intermediate checkpoints (samples and state of the chain) so you can resume or inspect progress.
Troubleshooting

If the example script fails due to a missing package: run julia --project=. -e 'using Pkg; Pkg.instantiate()' and then rerun the script.
If runtime is very long: reduce Niter or Nparticles to verify functionality, then scale up for production runs.
If posterior samples show poor mixing: try tuning the proposal covariance or increase Nparticles in the particle filter.
If plots do not render in non-interactive environments: either save plots to files (the scripts typically do this) or run Julia in an environment with a plotting backend that supports headless rendering (e.g., GR with PNG output).
