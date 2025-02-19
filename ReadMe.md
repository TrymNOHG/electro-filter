This repository contains a Python script that solves the incompressible Navier-Stokes equations using the **FEniCS** finite element library. The code demonstrates how to set up and solve a mixed formulation of the Navier-Stokes equations for fluid flow in a 2D unit square domain. This will further be used to solve the Nernst-Planck Equation for an electrochemical filter. 

## **What the Code Does (hopefully)**
1. Mesh Creation
2. Explores Function Spaces
3. Boundary Conditions
4. Weak Formulation using mixed FEM
5. Solver The system is solved using FEniCS's
6. Visualization plotted using `matplotlib`.

## **Dependencies**
To run this code, you need the following Python libraries:
- `numpy`
- `matplotlib`
- `fenics`

Setting Up FEniCS
If you're new to FEniCS, check out this introductory video on setting up the FEniCS environment: https://www.youtube.com/watch?v=ow-uGsAvGTQ

Not a video kind of guy? Check out the documentation: https://fenicsproject.org/documentation/
