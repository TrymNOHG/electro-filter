# Optimization of Operating Conditions for Electrochemical Water Filtration to Increase Efficiency

This repository explores **electrochemical water filtration systems** using a combination of numerical simulation techniques and optimization algorithms. The aim is to model, analyze, and optimize electrochemical water filters for both physical accuracy and operational efficiency.

## Simulation Methods

- **Finite Element Method (FEM):**  
  
- **Finite Difference Method (FDM):**  
  
- **Monte Carlo Simulation:**  
  Models particle-based transport and adsorption behaviors stochastically to evaluate probabilistic outcomes.

## Optimization Techniques

- **Bayesian Optimization:**  
  Applied to tune physical/operational parameters for a weighted sum of objectives.
  
- **NSGA-II (Non-dominated Sorting Genetic Algorithm II):**  
  Used for **multi-objective optimization**, e.g., maximizing sum of deposition while minimizing the variance of the deposition profile (i.e. maximizing uniformity of the deposition).


## Getting Started

### Prerequisites

Make sure you have Python 3.8+ installed. Recommended: set up a virtual environment.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Purpose of repository

The purpose of this repository is to model development of deposition from an electrochemical water filtration system. As such, various physical models were used to explore the adsorption and desorption dynamics such as the Langmuir model, Pouseuille flow, and Nernst-Planck equation. In order to tie this research back to more practical usage, both single and multiple objective optimization was performed using Bayesian optimization from Scikit-optimize and NSGA-2 from scratch, respectively. The resulting optima can be used to determine the trade-offs of certain system parameters in the presence of operational requirements.

## Further Work
- Increase numerical stability of FEM and FDM solvers.
- Look into optimal control for the system.
- Run experimentations to validate simulations.
