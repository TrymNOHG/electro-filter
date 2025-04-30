import numpy as np
import matplotlib.pyplot as plt
from ngsolve import Mesh, CoefficientFunction, IfPos

def MonteCarloHitDistribution(
    mesh,
    velocity_cf,
    E_cf,
    mu_e=1.0,
    nparticles=2000,
    dt=0.001,
    max_time=2.0,
    show_plot=True
):
    """
    Estimate where charged particles 'hit' the top wall y=1
    by a simple Monte-Carlo/particle-tracing approach.
    
    mesh         : ngsolve.Mesh
    velocity_cf  : ngsolve.CoefficientFunction, e.g. (u_x, u_y)
    E_cf         : ngsolve.CoefficientFunction, e.g. (E_x, E_y) 
                   but possibly zero outside an interval of x.
    mu_e         : electrophoretic mobility factor.
    nparticles   : number of particles to simulate.
    dt           : time-step for the ODE integration (simple Euler).
    max_time     : end time; particles not hitting y=1 by then are missed.
    show_plot    : if True, produce a histogram at the end.
    
    Returns
    -------
    hits_x : 1D numpy array of x-locations of top-wall hits, or NaN if missed.
    """

    # 1) Initial positions: all at x=0, y~Uniform(0..1)
    y0 = np.random.rand(nparticles)
    x0 = np.zeros(nparticles)

    # store current positions
    xs = x0.copy()
    ys = y0.copy()

    # final "hit" x-locations
    hits_x = np.full(nparticles, np.nan)

    nsteps = int(max_time / dt)

    for step in range(nsteps):
        active = np.isnan(hits_x)  # those not yet hitting the top
        if not np.any(active):
            break

        xa = xs[active]
        ya = ys[active]

        vx_vals = np.empty_like(xa)
        vy_vals = np.empty_like(xa)
        Ex_vals = np.empty_like(xa)
        Ey_vals = np.empty_like(xa)

        # Evaluate velocity and E for each particle
        for i in range(len(xa)):
            vx_vals[i], vy_vals[i] = velocity_cf(mesh(xa[i], ya[i]))
            Ex_vals[i], Ey_vals[i] = E_cf(mesh(xa[i], ya[i]))

        # Add drift = mu_e * E
        vx_vals += mu_e * Ex_vals
        vy_vals += mu_e * Ey_vals

        # Euler step
        xa_new = xa + dt * vx_vals
        ya_new = ya + dt * vy_vals

        # check crossing of y=1
        crossing = (ya < 1.0) & (ya_new >= 1.0)

        fraction = np.where(crossing,
                            (1.0 - ya) / (ya_new - ya),
                            0.0)
        x_hit = xa + fraction*(xa_new - xa)

        idx_active = np.where(active)[0]
        idx_cross = idx_active[crossing]
        hits_x[idx_cross] = x_hit[crossing]

        # update positions only for those not crossing
        xa[~crossing] = xa_new[~crossing]
        ya[~crossing] = ya_new[~crossing]
        xs[active] = xa
        ys[active] = ya

    if show_plot:
        valid = hits_x[~np.isnan(hits_x)]
        plt.figure()
        plt.hist(valid, bins=1000, range=(0,3))
        plt.xlabel("x-coordinate of top-wall hit")
        plt.ylabel("Number of particles hitting")
        plt.title("MonteCarlo hits on top boundary (y=1)")
        plt.grid(True)
        plt.savefig("10000_linear_test.png")

    return hits_x


if __name__ == "__main__":
    from netgen.geom2d import SplineGeometry

    # 1) Domain geometry: [0,3] x [0,1]
    geo = SplineGeometry()
    geo.AddRectangle((0,0),(3,1), bcs=["inflow","wall","outflow","wall"])
    mesh = Mesh(geo.GenerateMesh(maxh=0.02))

    # 2) Define velocity and piecewise E.  E=0 outside [0.5..2.5]
    from ngsolve import x, y

    #velocity_cf = CoefficientFunction((4*y*(1-y), 0))
    #velocity_cf = CoefficientFunction((1.0, 0.0))  # constant horizontal
    velocity_cf = CoefficientFunction((y, 0.0))  # linear
    # IfPos( x-0.5,   IfPos( 2.5-x, 1.0, 0.0 ),  0.0 )
    # that is 1.0 only if (x>=0.5 AND x<=2.5), else 0.0
    E_switch = IfPos(x - 0.5,
                     IfPos(2.5 - x, 1.0, 0.0),
                     0.0)

    # The field is in y-direction with magnitude E_switch
    E_cf = CoefficientFunction((0, E_switch))

    # 3) Run Monte-Carlo
    hits = MonteCarloHitDistribution(
        mesh=mesh,
        velocity_cf=velocity_cf,
        E_cf=E_cf,
        mu_e=0.5,          # scale factor for E
        nparticles=10000,
        dt=0.005,
        max_time=3.0,
        show_plot=True
    )
    print("Number of hits:", np.count_nonzero(~np.isnan(hits)))
    print("Example hits (first 20):", hits[:20])

