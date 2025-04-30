from ngsolve import *
from ngsolve.webgui import Draw

def simple_newton(gfu, a, tol=1e-13, maxiter=10):
    """
    Simple Newton iteration to solve a nonlinear system.
    
    Parameters:
      gfu     - GridFunction (solution vector, modified in place)
      a       - BilinearForm used to compute the residual and its linearization
      tol     - Convergence tolerance (based on a residual inner product)
      maxiter - Maximum number of iterations
    """ 
    res = gfu.vec.CreateVector()
    du = gfu.vec.CreateVector()
    fes = gfu.space
    for _ in range(maxiter):
        a.Apply(gfu.vec, res)
        a.AssembleLinearization(gfu.vec)
        du.data = a.mat.Inverse(fes.FreeDofs()) * res
        gfu.vec.data -= du
        # Stopping criterion: norm(du, res) small enough
        stopcrit = sqrt(abs(InnerProduct(du, res)))
        if stopcrit < tol:
            break


class NSStationary:
    """
    A class for solving the stationary Navier-Stokes equations
    using a Taylor-Hood finite element discretization.
    
    Attributes:
      nu            - kinematic viscosity
      no_slip       - string specifying boundary names on which Dirichlet (no-slip) conditions are imposed
      full_dirichlet- bool; if True, all boundary data are enforced strongly
      fes           - Finite element space (set by make_fem_space)
      a             - Bilinear form for the variational formulation
      l             - Linear form (if needed)
    """
    def __init__(self, nu, no_slip=None, full_dirichlet=False):
        self.nu = nu
        self.no_slip = no_slip
        self.full_dirichlet = full_dirichlet
        self.fes = None
        self.a = None
        self.l = None

    def make_fem_space(self, mesh, order=2):
        """
        Create the finite element space using Taylor-Hood elements.
        
        Parameters:
          mesh  - The computational mesh.
          order - Order for the velocity space; pressure is order-1.
        
        Returns:
          None (the fes attribute is set).
        """
        V = VectorH1(mesh, order=order, dirichlet=self.no_slip)
        Q = H1(mesh, order=order-1)
        self.fes = V * Q
        if self.full_dirichlet:
            # Enforce full Dirichlet conditions (e.g., by introducing a NumberSpace)
            N = NumberSpace(mesh)
            self.fes = self.fes * N
        return None

    def make_bilinear_form(self, f):
        """
        Assemble the bilinear form for the stationary Navier-Stokes equations.
        The form includes the viscous term, the convective term, and the pressure coupling.
        A body force term is added on the right-hand side.
        
        Parameters:
          f - CoefficientFunction representing the body force.
        
        Returns:
          None (the a attribute is set).
        """
        a = BilinearForm(self.fes)
        if self.full_dirichlet:
            (u, p, lam), (v, q, mu) = self.fes.TnT()
            a += (self.nu * InnerProduct(grad(u), grad(v))
                  + InnerProduct(grad(u) * u, v)
                  - div(u) * q - div(v) * p
                  + lam * q + mu * p) * dx
        else:
            (u, p), (v, q) = self.fes.TnT()
            a += (self.nu * InnerProduct(grad(u), grad(v))
                  + InnerProduct(grad(u) * u, v)
                  - div(u) * q - div(v) * p) * dx
        # Add forcing term (moved to the left-hand side as -f*v*dx)
        a += -f * v * dx
        self.a = a
        return None

    def neumann_condition(self, g, region):
        """
        Augment the bilinear form with a Neumann condition.
        
        Parameters:
          g      - CoefficientFunction for the Neumann data.
          region - A string (or similar) defining the boundary region.
        
        Returns:
          None (the bilinear form self.a is modified).
        """
        if self.full_dirichlet:
            return None
        (u, p), (v, q) = self.fes.TnT()
        self.a += g * v * ds(definedon=region)
        return None

    def solve(self, mesh, f=CF((0,0)), u_D=0,
              g_neu=None, neu_region=None, order=2,
              nonlinear_solver=simple_newton, tol=1e-13, maxiter=10):
        """
        Solve the stationary Navier-Stokes problem.
        
        Parameters:
          mesh              - Computational mesh.
          f                 - Body force as a CoefficientFunction.
          u_D               - Dirichlet data for the velocity field.
          g_neu             - Neumann data (optional).
          neu_region        - Boundary region for Neumann data (optional).
          order             - Finite element order.
          nonlinear_solver  - Function implementing the nonlinear solver (default: simple_newton).
          tol, maxiter      - Tolerance and maximum iterations for the nonlinear solver.
        
        Returns:
          gfu - GridFunction containing the computed solution.
        """
        self.make_fem_space(mesh, order)
        self.make_bilinear_form(f)
        if neu_region is not None:
            self.neumann_condition(g_neu, neu_region)
        gfu = GridFunction(self.fes)
        # Set Dirichlet data on the designated boundary if provided.
        if self.no_slip is not None:
            gfu.components[0].Set(u_D, definedon=mesh.Boundaries(self.no_slip))
        nonlinear_solver(gfu, self.a, tol=tol, maxiter=maxiter)
        return gfu


def calc_numerical_quantities(u_h, p_h, mesh, nu=1e-3, order=2, u_mean=1.0, L=0.1):
    """
    Compute some numerical quantities (e.g. drag and lift coefficients, pressure difference)
    based on the computed solution.
    
    Parameters:
      u_h   - Velocity GridFunction.
      p_h   - Pressure GridFunction.
      mesh  - Computational mesh.
      nu    - Viscosity.
      order - Order for a temporary finite element space.
      u_mean- Mean velocity (for normalization).
      L     - Characteristic length.
      
    Returns:
      A tuple (C_D, C_L, p_diff) where:
         C_D    - Drag coefficient.
         C_L    - Lift coefficient.
         p_diff - Pressure difference computed at two specific points.
    """
    n = specialcf.normal(2)
    tang = specialcf.tangential(2)
    # Create a temporary H1 space for computing derived quantities.
    fes_calc = H1(mesh, order=order)
    u_h1 = GridFunction(fes_calc)
    u_h1.Set(u_h * CF((1, 0)))
    u_h2 = GridFunction(fes_calc)
    u_h2.Set(u_h * CF((0, 1)))
    F_D = Integrate(
        (BoundaryFromVolumeCF(nu * CF((grad(u_h1) * n, grad(u_h2) * n))
                              * tang * (n * CF((0,1))))
         - p_h * (n * CF((1,0)))),
        mesh, VOL_or_BND=BND, definedon=mesh.Boundaries('cyl')
    )
    F_L = -Integrate(
        (BoundaryFromVolumeCF(nu * CF((grad(u_h1) * n, grad(u_h2) * n))
                              * tang * (n * CF((1,0))))
         + p_h * (n * CF((0,1)))),
        mesh, VOL_or_BND=BND, definedon=mesh.Boundaries('cyl')
    )
    C_D = 2 / (u_mean**2 * L) * F_D
    C_L = 2 / (u_mean**2 * L) * F_L
    p_diff = p_h(mesh(0.15, 0.2)) - p_h(mesh(0.25, 0.2))
    return C_D, C_L, p_diff


# --- Time-Stepping Solvers ---
# The following functions implement BDF1 and BDF2 solvers
# in either an IMEX or a Newton formulation.

def BDF1_solver_IMEX(mesh, fes, u_0=None, u_D=lambda t: CF((0,0)),
                     f=lambda t: CF((0,0)), nu=1, dt=1, T=1,
                     no_slip=None, full_dirichlet=False,
                     g=lambda t: CF((0,0)), do_nothing=None,
                     tol=1e-11, maxiter=30, calc_coeff=False):
    """
    First-order Backward Differentiation Formula (BDF1) solver using an IMEX approach.
    
    Parameters:
      mesh, fes     - Mesh and finite element space.
      u_0           - Initial condition (if None, computed from u_D(0)).
      u_D           - Dirichlet data as a function of time.
      f             - Forcing function as a function of time.
      nu, dt, T     - Viscosity, time step, and final time.
      no_slip       - Boundary (name) on which to apply Dirichlet conditions.
      full_dirichlet- Boolean flag for full Dirichlet conditions.
      g, do_nothing - Additional boundary forcing functions.
      tol, maxiter  - Tolerance and max iterations for the nonlinear solver.
      calc_coeff    - If True, compute additional coefficients.
      
    Returns:
      gfut - A GridFunction containing the time evolution of the solution.
    """
    if full_dirichlet:
        (u, p, lam), (v, q, mu) = fes.TnT()
        stokes = (nu * InnerProduct(grad(u), grad(v))
                  - div(u) * q - div(v) * p + lam * q + mu * p)
    else:
        (u, p), (v, q) = fes.TnT()
        stokes = (nu * InnerProduct(grad(u), grad(v))
                  - div(u) * q - div(v) * p)
    gfu = GridFunction(fes)
    gfu0 = GridFunction(fes)
    if u_0 is None:
        gfu0.components[0].Set(u_D(0), definedon=mesh.Boundaries(no_slip))
        a = BilinearForm(fes)
        a += stokes * dx
        a.Assemble()
        l = LinearForm(fes)
        if do_nothing is not None:
            l += -g(0) * v * ds(definedon=do_nothing)
        l.Assemble()
        res = l.vec - a.mat * gfu0.vec
        inv_stokes = a.mat.Inverse(fes.FreeDofs())
        gfu0.vec.data += inv_stokes * res
    else:
        gfu0.components[0].Set(u_0)
    gfut = GridFunction(fes, multidim=0)
    gfut.AddMultiDimComponent(gfu0.vec)
    m = BilinearForm(fes)
    m += u * v * dx
    m.Assemble()
    mstar = BilinearForm(fes)
    mstar += InnerProduct(u, v) * dx + dt * stokes * dx
    mstar.Assemble()
    inv = mstar.mat.Inverse(fes.FreeDofs())
    conv = BilinearForm(fes, nonassemble=True)
    conv += (Grad(u) * u) * v * dx
    t = 0
    while t < T:
        t += dt
        l = LinearForm(fes)
        l += dt * f(t) * v * dx
        if do_nothing is not None:
            l += -dt * g(t) * v * ds(definedon=do_nothing)
        l.Assemble()
        conv.Assemble()
        res = l.vec + (m.mat - dt * conv.mat) * gfu0.vec
        gfu.components[0].Set(u_D(t), definedon=mesh.Boundaries(no_slip))
        gfu.vec.data += inv * (res - mstar.mat * gfu.vec)
        gfut.AddMultiDimComponent(gfu.vec)
        gfu0.vec.data = gfu.vec
    return gfut

def BDF1_solver_Newton(mesh, fes, u_0=None, u_D=lambda t: CF((0,0)),
                       f=lambda t: CF((0,0)), nu=1, dt=1, T=1,
                       no_slip=None, full_dirichlet=False,
                       g=lambda t: CF((0,0)), do_nothing=None,
                       tol=1e-11, maxiter=30, calc_coeff=False):
    """
    BDF1 solver using Newton’s method.
    
    Parameters are similar to BDF1_solver_IMEX.
    
    Returns:
      gfut - GridFunction (time evolution of the solution).
    """
    if full_dirichlet:
        (u, p, lam), (v, q, mu) = fes.TnT()
        stokes = (nu * InnerProduct(grad(u), grad(v))
                  - div(u) * q - div(v) * p + lam * q + mu * p)
    else:
        (u, p), (v, q) = fes.TnT()
        stokes = (nu * InnerProduct(grad(u), grad(v))
                  - div(u) * q - div(v) * p)
    gfu = GridFunction(fes)
    gfu0 = GridFunction(fes)
    if u_0 is None:
        gfu0.components[0].Set(u_D(0), definedon=mesh.Boundaries(no_slip))
        a = BilinearForm(fes)
        a += stokes * dx
        a.Assemble()
        l = LinearForm(fes)
        if do_nothing is not None:
            l += -g(0) * v * ds(definedon=do_nothing)
        l.Assemble()
        res = l.vec - a.mat * gfu0.vec
        inv_stokes = a.mat.Inverse(fes.FreeDofs())
        gfu0.vec.data += inv_stokes * res
    else:
        gfu0.components[0].Set(u_0)
    gfut = GridFunction(gfu.space, multidim=0)
    gfut.AddMultiDimComponent(gfu0.vec)
    if calc_coeff:
        coeff_list = []
    t = 0
    while t < T:
        t += dt
        A = BilinearForm(fes)
        A += u * v * dx
        A += dt * stokes * dx
        A += dt * InnerProduct(grad(u) * u, v) * dx
        A += -dt * f(t) * v * dx
        A += -gfu0.components[0] * v * dx
        if do_nothing is not None:
            A += dt * g(t) * v * ds(definedon=do_nothing)
        gfu.components[0].Set(u_D(t), definedon=mesh.Boundaries(no_slip))
        simple_newton(gfu, A, tol=tol, maxiter=maxiter)
        gfut.AddMultiDimComponent(gfu.vec)
        gfu0.vec.data = gfu.vec
        if calc_coeff:
            u_h, p_h = gfu.components[:2]
            coeff_list.append(calc_numerical_quantities(u_h, p_h, mesh, nu, 3))
    if calc_coeff:
        return gfut, coeff_list
    return gfut

def BDF2_solver_IMEX(mesh, fes, u_0=None, u_D=lambda t: CF((0,0)),
                     f=lambda t: CF((0,0)), nu=1, dt=1, T=1,
                     no_slip=None, full_dirichlet=False,
                     g=lambda t: CF((0,0)), do_nothing=None,
                     tol=1e-11, maxiter=30, calc_coeff=False):
    """
    Second-order Backward Differentiation Formula (BDF2) solver (IMEX variant).
    
    Parameters are similar to the BDF1 solvers.
    
    Returns:
      gfut - GridFunction (solution time evolution).
    """
    if full_dirichlet:
        (u, p, lam), (v, q, mu) = fes.TnT()
        stokes = (nu * InnerProduct(grad(u), grad(v))
                  - div(u) * q - div(v) * p + lam * q + mu * p)
    else:
        (u, p), (v, q) = fes.TnT()
        stokes = (nu * InnerProduct(grad(u), grad(v))
                  - div(u) * q - div(v) * p)
    gfu = GridFunction(fes)
    gfu0 = GridFunction(fes)
    if u_0 is None:
        gfu0.components[0].Set(u_D(0), definedon=mesh.Boundaries(no_slip))
        a = BilinearForm(fes)
        a += stokes * dx
        a.Assemble()
        l = LinearForm(fes)
        if do_nothing is not None:
            l += -g(0) * v * ds(definedon=do_nothing)
        l.Assemble()
        res = l.vec - a.mat * gfu0.vec
        inv_stokes = a.mat.Inverse(fes.FreeDofs())
        gfu0.vec.data += inv_stokes * res
    else:
        gfu0.components[0].Set(u_0)
    gfut = GridFunction(fes, multidim=0)
    gfut.AddMultiDimComponent(gfu0.vec)
    m = BilinearForm(fes)
    m += u * v * dx
    m.Assemble()
    conv = BilinearForm(fes, nonassemble=True)
    conv += (Grad(u) * u) * v * dx
    t = 0
    # One step of BDF1:
    mstar = BilinearForm(fes)
    mstar += InnerProduct(u, v) * dx + dt * stokes * dx
    mstar.Assemble()
    inv = mstar.mat.Inverse(fes.FreeDofs())
    t += dt
    l = LinearForm(fes)
    l += dt * f(t) * v * dx
    if do_nothing is not None:
        l += -dt * g(t) * v * ds(definedon=do_nothing)
    l.Assemble()
    conv.Assemble()
    res = l.vec + (m.mat - dt * conv.mat) * gfu0.vec
    gfu.components[0].Set(u_D(t), definedon=mesh.Boundaries(no_slip))
    gfu.vec.data += inv * (res - mstar.mat * gfu.vec)
    gfut.AddMultiDimComponent(gfu.vec)
    gfu1 = GridFunction(fes)
    gfu1.vec.data = gfu.vec
    mstar = BilinearForm(fes)
    mstar += InnerProduct(u, v) * dx + 2/3 * dt * stokes * dx
    mstar.Assemble()
    inv = mstar.mat.Inverse(fes.FreeDofs())
    while t < T:
        t += dt
        l = LinearForm(fes)
        l += 2/3 * dt * f(t) * v * dx
        if do_nothing is not None:
            l += -2/3 * dt * g(t) * v * ds(definedon=do_nothing)
        l.Assemble()
        conv.Assemble()
        res = l.vec + (4/3 * m.mat - 2/3 * dt * conv.mat) * gfu1.vec - 1/3 * m.mat * gfu0.vec
        gfu.components[0].Set(u_D(t), definedon=mesh.Boundaries(no_slip))
        gfu.vec.data += inv * (res - mstar.mat * gfu.vec)
        gfut.AddMultiDimComponent(gfu.vec)
        gfu0.vec.data = gfu1.vec
        gfu1.vec.data = gfu.vec
    return gfut

def BDF2_solver_Newton(mesh, fes, u_0=None, u_D=lambda t: CF((0,0)),
                       f=lambda t: CF((0,0)), nu=1, dt=1, T=1,
                       no_slip=None, full_dirichlet=False,
                       g=lambda t: CF((0,0)), do_nothing=None,
                       tol=1e-11, maxiter=30, calc_coeff=False):
    """
    Second-order BDF solver using Newton's method.
    
    Parameters are similar to the other solvers.
    
    Returns:
      gfut - GridFunction containing the solution evolution.
    """
    if full_dirichlet:
        (u, p, lam), (v, q, mu) = fes.TnT()
        stokes = (nu * InnerProduct(grad(u), grad(v))
                  - div(u) * q - div(v) * p + lam * q + mu * p)
    else:
        (u, p), (v, q) = fes.TnT()
        stokes = (nu * InnerProduct(grad(u), grad(v))
                  - div(u) * q - div(v) * p)
    gfu = GridFunction(fes)
    gfu0 = GridFunction(fes)
    if u_0 is None:
        gfu0.components[0].Set(u_D(0), definedon=mesh.Boundaries(no_slip))
        a = BilinearForm(fes)
        a += stokes * dx
        a.Assemble()
        l = LinearForm(fes)
        if do_nothing is not None:
            l += -g(0) * v * ds(definedon=do_nothing)
        l.Assemble()
        res = l.vec - a.mat * gfu0.vec
        inv_stokes = a.mat.Inverse(fes.FreeDofs())
        gfu0.vec.data += inv_stokes * res
    else:
        gfu0.components[0].Set(u_0)
    gfut = GridFunction(gfu.space, multidim=0)
    gfut.AddMultiDimComponent(gfu0.vec)
    if calc_coeff:
        coeff_list = []
    t = 0
    while t < T:
        t += dt
        A = BilinearForm(fes)
        A += u * v * dx
        A += dt * stokes * dx
        A += dt * InnerProduct(grad(u) * u, v) * dx
        A += -dt * f(t) * v * dx
        A += -gfu0.components[0] * v * dx
        if do_nothing is not None:
            A += dt * g(t) * v * ds(definedon=do_nothing)
        gfu.components[0].Set(u_D(t), definedon=mesh.Boundaries(no_slip))
        simple_newton(gfu, A, tol=tol, maxiter=maxiter)
        gfut.AddMultiDimComponent(gfu.vec)
        gfu0.vec.data = gfu.vec
        if calc_coeff:
            u_h, p_h = gfu.components[:2]
            coeff_list.append(calc_numerical_quantities(u_h, p_h, mesh, nu, 3))
    if calc_coeff:
        return gfut, coeff_list
    return gfut


def NS_Solver(mesh, fes, u_0=None, 
              u_D=lambda t: CF((0,0)), 
              f=lambda t: CF((0,0)), 
              nu=1, dt=1, T=1,
              no_slip=None, full_dirichlet=False,
              g=lambda t: CF((0,0)), do_nothing=None,
              tol=1e-11, maxiter=30, 
              method='BDF1_NEWTON', calc_coeff=False):
    """
    Solve the Navier–Stokes problem using one of the available time-stepping solvers.
    
    Parameters:
      mesh         : Mesh
                     The computational mesh.
      fes          : FESpace
                     The finite element space for the problem.
      u_0          : GridFunction, optional
                     Initial condition for the solution. If None, the solver initializes
                     using u_D(0) on the no-slip boundary.
      u_D          : callable, optional
                     A function of time returning the Dirichlet data as a CoefficientFunction.
      f            : callable, optional
                     A function of time returning the volume force (as a CF).
      nu           : float, optional
                     Kinematic viscosity.
      dt           : float, optional
                     Time step size.
      T            : float, optional
                     Final time.
      no_slip      : str, optional
                     Name of the boundary on which to impose Dirichlet (no-slip) conditions.
      full_dirichlet: bool, optional
                     If True, impose Dirichlet data on all boundaries (using a full Dirichlet formulation).
      g            : callable, optional
                     Additional boundary force as a function of time.
      do_nothing   : str or region specifier, optional
                     Region on which no additional forcing is applied.
      tol          : float, optional
                     Tolerance for the nonlinear solver.
      maxiter      : int, optional
                     Maximum iterations for the nonlinear solver.
      method       : str, optional
                     Time-stepping method to use. Options are:
                     'BDF1_NEWTON', 'BDF2_NEWTON', 'BDF1_IMEX', 'BDF2_IMEX'.
      calc_coeff   : bool, optional
                     If True, compute additional coefficients during the time stepping.
    
    Returns:
      gfut         : GridFunction
                     A GridFunction containing the time evolution of the solution.
    
    Raises:
      ValueError   : if the specified method is not recognized.
    """
    solvers = {
        'BDF1_NEWTON': BDF1_solver_Newton,
        'BDF2_NEWTON': BDF2_solver_Newton,
        'BDF1_IMEX':   BDF1_solver_IMEX,
        'BDF2_IMEX':   BDF2_solver_IMEX
    }
    solver_key = method.upper()
    solver_func = solvers.get(solver_key)
    if solver_func is None:
        raise ValueError(f"Method '{method}' not recognized. Valid methods are: {list(solvers.keys())}")
    
    return solver_func(
        mesh, fes,
        u_0=u_0,
        u_D=u_D,
        f=f,
        nu=nu,
        dt=dt,
        T=T,
        no_slip=no_slip,
        full_dirichlet=full_dirichlet,
        g=g,
        do_nothing=do_nothing,
        tol=tol,
        maxiter=maxiter,
        calc_coeff=calc_coeff
    )
