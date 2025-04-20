# Necessary libraries and useful parameters
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.colorbar as cb

import scipy.sparse as sp
import warnings
warnings.simplefilter("ignore", RuntimeWarning)

newparams = {'font.family': 'cmr10', 'mathtext.fontset': 'cm',
             'axes.grid': False, 'axes.labelsize': 22,
             'xtick.labelsize': 18, 'ytick.labelsize': 18,
             'legend.fontsize': 18, 'axes.titlesize': 20,
             'figure.figsize': (9,6), 'lines.linewidth': 2.5,
             'axes.formatter.use_mathtext': True}
plt.rcParams.update(newparams)

# Color maps
concentration_colors = ['#00274D', '#1B4F72', '#CFA6A6', '#CD5C5C', '#8B3A3A', '#641010']
concentration_cmap = mcolors.LinearSegmentedColormap.from_list('custom_concentrations', concentration_colors)

deposition_colors = ['#F6CCCC', '#CD5C5C', '#641010']
deposition_cmap = mcolors.LinearSegmentedColormap.from_list('custom_reds', deposition_colors)

# Model dimensions
dims = ((0, 1), (0, 1), (0, 1))

def domain(MK, dims=dims):
    M, K = MK
    h, ht = 1/M, 1/K

    z = np.linspace(dims[0][0], dims[0][1], M+1)
    r = np.linspace(dims[1][0], dims[1][1], M+1)
    t = np.linspace(dims[2][0], dims[2][1], K+1)

    ZZ, RR = np.meshgrid(z, r) 

    return (h, ht), (z, r, t), (ZZ, RR)

# Scaled block matrix coefficients
def matrix_coeffs(model, params):
    ht, h, r = model
    R, L, E, U, D = params

    δ = ht/h**2 * D
    γ = δ * (R/L)**2
    ξ = ht/h * D*(E-1)/r
    𝜗 = ht/h * U*(1-r**2)
    ω = ξ + 𝜗 + 2*(γ + δ) + 1

    return δ, γ, ξ, 𝜗, ω



def S(M, r, z, t, params):
    R, L, E, U, D = params
    
    term_time = -np.sin(np.pi*r) * np.cos(np.pi*z)
    term_diff = D * np.pi**2 * np.sin(np.pi*r) * np.cos(np.pi*z) * (1 + (R/L)**2)
    term_radial = D * (E - 1) * (1/r) * np.pi * np.cos(np.pi*r) * np.cos(np.pi*z)
    term_conv = U * (1 - r**2) * (-np.pi * np.sin(np.pi*z) * np.sin(np.pi*r))

    return (term_time + term_diff + term_radial + term_conv) * np.exp(-t)

def coeff_matrix_ms(M:int, coeffs) -> sp.csr_matrix:
    δ, γ, ξ, 𝜗, ω = coeffs
    Z = (M-1)**2

    ll = np.repeat(δ + ξ[1:-1], M-1)      # Lowest diag
    l = np.repeat(γ + 𝜗[1:-1], M-1)       # Next lowest diag
    d = np.repeat(ω[1:-1], M-1)            # Main diag 
    u = np.full(Z, γ)                    # Next highest diag
    uu = np.full(Z, δ)                   # Highest diag
    
    l[M-2::M-1], u[M-2::M-1] = 0, 0 # Every M-1'th element is zero in l and d
    
    A = sp.diags([ll, l, -d, u, uu], [-(M-1), -1, 0, 1, (M-1)], (Z, Z), format='csr')

    return A

def boundary_vector_ms(M, model:tuple, coeffs, k:int, Ck:np.ndarray, BC:tuple) -> np.ndarray:
  z, r, t = model
  δ, γ, ξ, 𝜗, ω = coeffs
  g1, g2, g3, g4 = BC
  
  # Construct a matrix containing boundary contributions.
  G = np.zeros((M-1, M-1))
  G[::-1, 0] += (γ+𝜗[1:-1]) * g1(r[1:-1], t[k+1]) # Left contribution
  G[-1, :] += (δ+ξ[1]) * g2(z[1:-1], t[k+1])              # Bottom contribution
  G[0, :] += δ * g3(z[1:-1], t[k+1])                   # Top contribution
  G[::-1, -1] += γ * g4(r[1:-1], t[k+1])

  return G[::-1].ravel() # return G as a vector

def concentration_scheme_ms(M, model, params, A:np.ndarray, g:np.ndarray, k:int, Ck:np.ndarray, BC:tuple, S) -> np.ndarray:
    (h, ht), (z, r, t), (ZZ, RR) = model
    g1, g2, g3, g4 = BC
   
    C = np.zeros((M+1, M+1))
    C[::-1, 0] = g1(r, t[k+1]) # Left boundary
    C[-1, :] = g2(z, t[k+1])   # Bottom boundary  
    C[0, 1:] = g3(z[1:-1], t[k+1]) # Top boundary
    C[::-1, -1] = g4(r, t[k+1])
    
    C_interior = sp.linalg.spsolve(A, -(g + Ck[1:-1, 1:-1][::-1].ravel() + ht*S(M, RR, ZZ, t[k+1], params)[1:-1, 1:-1][::-1].ravel()))
    C[1:-1, 1:-1] = np.reshape(C_interior, (M-1, M-1))
   
    return C


# Initial condition
def f_ms(r, z):                   
    return np.sin(np.pi*r)*np.cos(np.pi*z)   
                                                         
# Boundary conditions
def i_ms(r, t): # inlet
    return np.sin(np.pi*r)*np.exp(-t)

def e_ms(z, t): # electric
    return 0

def d_ms(z, t): # top
    return 0

def o_ms(r, t):
    return -np.sin(np.pi*r)*np.exp(-t)

def c_exact(r, z, t):
    return np.sin(np.pi*r)*np.cos(np.pi*z)*np.exp(-t)


def spatial_converge(params, M_list, K):
    R, L, E, U, D = params
    BC = (i_ms, e_ms, d_ms, o_ms)

    errors_Linf , errors_L2 = np.zeros(len(M_list)), np.zeros(len(M_list))
    H = np.zeros(len(M_list))
    ht = 1/K
    for j, M in enumerate(M_list):
        h = 1/M
        (h, ht), (z, r, t), (ZZ, RR) = domain((M, K))

        coeffs = matrix_coeffs((ht, h, r), (R, L, E, U, D))
        A = coeff_matrix_ms(M, coeffs)

        Ck = f_ms(RR, ZZ)
        for k in range(K-1):
            g = boundary_vector_ms(M, (z, r, t), coeffs, k, Ck, BC)
            Ck = concentration_scheme_ms(M, ((h, ht), (z, r, t), (ZZ, RR)), (R, L, E, U, D), A, g, k, Ck, BC, S)

        C_exact = c_exact(RR, ZZ, 1)
        error = C_exact - Ck

        errors_Linf[j] = np.linalg.norm(error, ord=np.inf)
        errors_L2[j] = np.sqrt(h) * np.linalg.norm(error, ord='fro')
        H[j] = h

    p_Linf = np.polyfit(np.log(H), np.log(errors_Linf), 1)[0]
    p_L2 = np.polyfit(np.log(H), np.log(errors_L2), 1)[0]

    return H, errors_Linf, errors_L2, p_Linf, p_L2

def temporal_converge(params, K_list, M):
    R, L, E, U, D = params
    BC = (i_ms, e_ms, d_ms, o_ms)

    errors_Linf , errors_L2 = np.zeros(len(K_list)), np.zeros(len(K_list))
    Ht = np.zeros(len(K_list))
    h = 1/M
    for j, K in enumerate(K_list):
        ht = 1/K
        (h, ht), (z, r, t), (ZZ, RR) = domain((M, K))

        coeffs = matrix_coeffs((ht, h, r), (R, L, E, U, D))
        A = coeff_matrix_ms(M, coeffs)

        Ck = f_ms(RR, ZZ)
        for k in range(K-1):  # loop through time
            g = boundary_vector_ms(M, (z, r, t), coeffs, k, Ck, BC)
            Ck = concentration_scheme_ms(M, ((h, ht), (z, r, t), (ZZ, RR)), (R, L, E, U, D), A, g, k, Ck, BC, S)

        C_exact = c_exact(RR, ZZ, 1)
        error = C_exact - Ck

        errors_Linf[j] = np.linalg.norm(error, ord=np.inf)
        errors_L2[j] = np.sqrt(h) * np.linalg.norm(error, ord='fro')
        Ht[j] = ht
        

    p_Linf = np.polyfit(np.log(Ht), np.log(errors_Linf), 1)[0]
    p_L2 = np.polyfit(np.log(Ht), np.log(errors_L2), 1)[0]

    return Ht, errors_Linf, errors_L2, p_Linf, p_L2

def plot_convergence(converge_func, params, resolution_list, fixed_resolution):
    # Call the appropriate convergence function
    H, errors_Linf, errors_L2, p_Linf, p_L2 = converge_func(params, resolution_list, fixed_resolution)

    # Determine label base depending on which convergence function is being used
    label_base = "h" if converge_func.__name__ == 'spatial_converge' else 'h_t'

    plt.loglog(H, errors_Linf, 'o-', c='#1B4F72', label=f'$p^{{L_\\infty}}_{{{label_base}}} = {p_Linf:.2f}$')
    plt.loglog(H, errors_L2, 'o-', c='#8B3A3A', label=f'$p^{{L_2}}_{{{label_base}}} = {p_L2:.2f}$')
    plt.xlabel(f'${label_base}$')
    plt.ylabel(f'$||e({label_base})||$')
    plt.legend()
    plt.grid(True)
    plt.show()


# Example 1: Naive parameters, naive grid refinements
R, L, E, U, D = 0.1, 1, 3, 4, 1
plot_convergence(spatial_converge, (R, L, E, U, D), [50, 100, 200, 400], 400)
plot_convergence(temporal_converge, (R, L, E, U, D), [50, 100, 200, 400], 400)

# Example 2: Optimized parameters, naive grid refinements
R, L, E, U, D = 3, 2, 9, 20, 0.0125 ## FIND PARAMETERS ##
plot_convergence(spatial_converge, (R, L, E, U, D), [50, 100, 200, 400], 400)

R, L, E, U, D = 3, 2, 9, 20, 0.0125 ## FIND PARAMETERS ##
plot_convergence(temporal_converge, (R, L, E, U, D), [50, 100, 200, 400], 400)

# Example 3: Optimized parameters, better grid refinements
R, L, E, U, D = 3, 2, 9, 20, 0.0125 ## FIND PARAMETERS ##
plot_convergence(spatial_converge, (R, L, E, U, D), [200, 400, 800, 1600, 3200], 4000)

R, L, E, U, D = 3, 2, 9, 20, 0.0125 ## FIND PARAMETERS ##
plot_convergence(temporal_converge, (R, L, E, U, D), [200, 400, 800, 1600, 3200], 4000)