import numpy as np
from scipy.linalg import eigh # Import for solving the generalized eigenvalue problem (for Hermitian *symmetric* matrices)
from scipy.special import jn_zeros, j0, j1, jn # jn for m > 0 Bessel functions
from scipy.ndimage import gaussian_filter1d

# -------------------- Azimuthal FFT parts --------------------
def theta_fft_components(Q, axisTHETA=1):
    """Return A0, A(m>=1), B(m>=1) for Q(z,theta,r)."""
    Nz, Nth, Nr = Q.shape # Shape of the input variable
    F = np.fft.rfft(Q, axis=axisTHETA) # Perform FFT on the azimuthal axis
    M = F.shape[axisTHETA]-1 # We only need the m>=1 components, so the shape is half of the original array
    A0 = F[:,0,:].real / Nth # Amplitude (0th components)
    scale = 2.0/Nth 
    A = np.zeros((M, Nr, Nz)); B = np.zeros((M, Nr, Nz))
    for m in range(1, M+1):
        Fm = np.transpose(F[:,m,:])
        A[m-1] = scale*Fm.real # Re(FFT) = Nth/2 * Am
        B[m-1] = -scale*Fm.imag # Im(FFT) = -Nth/2 * Bm
    return A0, A, B

def theta_ifft_components(A0, A, B, Ntheta):
    """
    Inverse Fourier Transform to retrieve the orignal signal
    """
    A0 = A0.transpose() #FFT outputs is (z,r), change it to (r,z)
    Nr, Nz = A0.shape 
    M = A.shape[0]
    th = np.linspace(0, 2*np.pi, Ntheta, endpoint=False)
    Q = np.zeros((Nr,Nz,Ntheta))
    Q += A0[:,:,None]
    for m in range(1, M+1):
        Q += A[m-1][:,:,None]*np.cos(m*th)[None,None,:]
        Q += B[m-1][:,:,None]*np.sin(m*th)[None,None,:]
    return Q

# ---------------------------------------------------------------------
# Quadrature weights
# ---------------------------------------------------------------------
def trapezoid_weights(x):
    """
    Computes trapezoidal rule weights for a given coordinate array x.
    These weights (W_i) are used for numerical integration: ∫ f(x) dx ≈ Σ f(x_i) * W_i.
    
    The function handles non-uniform grid spacing.
    """
    x = np.asarray(x)
    # Ensure the coordinate array is ascending for correct weight calculation.
    if np.any(np.diff(x) < 0):  # if decreasing
        x = x[::-1]  # flip to ascending
    dx = np.diff(x) # Segment lengths
    w = np.zeros_like(x, dtype=float)
    
    # Interior points: The weight is the average of the adjacent segment lengths.
    # W_i = 0.5 * (dx_{i-1} + dx_i)
    w[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    
    # End points: Half of the adjacent segment.
    w[0] = 0.5 * dx[0]
    w[-1] = 0.5 * dx[-1]
    return w
    
# ---------------------------------------------------------------------
# Fourier–Bessel radial modes
# ---------------------------------------------------------------------
def radial_modes(r, a=None, m=0, nmax=6, bc="dirichlet"):
    """
    Compute normalized Fourier–Bessel radial modes (J_m(k_mn * r))
    for azimuthal wavenumber m. The modes are orthonormal under the weight (r dr).

    Parameters
    ----------
    r : ndarray
        Radial coordinates (should be dimensionless, e.g., in [0, 1]).
    a : float
        Outer radius; if None, a = r.max(). Should be 1.0 if r is normalized.
    m : int
        Azimuthal wavenumber (order of Bessel function J_m).
    nmax : int
        Number of radial modes (n, 1 to nmax).
    bc : str
        'dirichlet' (J_m(k_mn*a)=0, zero value at boundary) or
        'neumann' (J_m'(k_mn*a)=0, zero derivative at boundary).

    Returns
    -------
    Phi : ndarray
        (Nr, nmax) orthonormal basis: Phi_n(r)
    k_mn : ndarray
        Radial wavenumbers (zeros / a)
    Wr : ndarray
        Integration weights (r dr) used for normalization
    """
    if a is None:
        a = r.max()

    # Determine the zeros (k*a) based on the boundary condition (BC)
    if bc == "dirichlet":
        # Dirichlet BC: J_m(k*a) = 0. Uses zeros of the Bessel function J_m itself.
        zeros = jn_zeros(m, nmax)
    elif bc == "neumann":
        # Neumann BC: J_m'(k*a) = 0. For m=0, J_0' = -J_1, so it uses J_1 zeros.
        # Generally, it simplifies to the zeros of J_{m+1} for our application.
        zeros = jn_zeros(m + 1, nmax)
    else:
        raise ValueError("bc must be 'dirichlet' or 'neumann'")

    # Calculate the wavenumbers: k_mn = zero / a
    k_mn = zeros / a
    
    # Create the raw Bessel function matrix: J_m(k*r)
    if m == 0:
        Phi_raw = j0(np.outer(r, k_mn)) # J0 for m=0
    else:
        Phi_raw = jn(m, np.outer(r, k_mn)) # Jm for m>0

    # Calculate the integration weight for cylindrical coordinates: Wr = r dr
    # This includes the Jacobian factor 'r' required for integration over a disk.
    Wr = trapezoid_weights(r) * r 

    # Normalize: Enforce the orthonormality condition: ∫_0^a Phi_n^2 * r dr = 1
    # This ensures that the basis functions form a proper orthonormal set.
    Phi_norm = np.sqrt(np.sum(Wr[:, None] * Phi_raw**2, axis=0))
    # Handle potential division by zero for the normalization factor
    Phi_norm[Phi_norm < 1e-12] = 1.0

    Phi = Phi_raw / Phi_norm
    return Phi, k_mn, Wr

# ---------------------------------------------------------------------
# Vertical modes (pressure coordinate) - Simplified Operator
# ---------------------------------------------------------------------
def vertical_modes_pressure(p, T=None, jmax=4, bc="neumann", smooth_sigma=0.0):
    """
    Compute orthonormal vertical modes ψ_j(p) in pressure coordinates.
    Solves a simplified, isothermal/isobaric-stability eigenvalue problem:
    -d/dp(S dψ/dp) = λ ψ, where S is a stability weight (S=1 for isothermal).
    This is solved using a finite-difference discretization of the operator.

    Parameters
    ----------
    p : ndarray
        Pressure levels, increasing downward. (Can be in Pa, hPa, or dimensionless)
    T : ndarray or None
        Temperature profile. If None, assumes isothermal (S=1).
    jmax : int
        Number of modes to retain.
    bc : str
        'dirichlet' (ψ=0 at boundaries) or 'neumann' (dψ/dp=0 at boundaries).
    smooth_sigma : float
        Optional Gaussian smoothing for T before derivative.
    """

    p = np.asarray(p)
    Np = len(p)
    Wp = trapezoid_weights(p) # Wp is 'dp' weights for the numerical integration (normalization)

    # --- Stability weight S (dimensionless)
    if T is None:
        Sp = np.ones_like(p) # Isothermal case (S=1), uniform stability
    else:
        T = np.asarray(T)
        if smooth_sigma > 0:
            T = gaussian_filter1d(T, smooth_sigma)
        # Scale T to create a dimensionless stability weight S proportional to temperature.
        Sp = T / T.max()

    # --- Finite-difference approximation for the operator -d/dp(S dψ/dp) ---
    A = np.zeros((Np, Np))
    for i in range(1, Np - 1):
        # Calculate pressure differences for non-uniform grid
        dp1, dp2 = p[i] - p[i - 1], p[i + 1] - p[i]
        
        # Discretization of the operator -d/dp(S dψ/dp) at internal node i
        # The equation is approximated as: (S_{i-1/2} * (ψ_i - ψ_{i-1}) / dp1 - S_{i+1/2} * (ψ_{i+1} - ψ_i) / dp2) / dp
        # We use S at the full levels (Sp[i-1] and Sp[i]) as an approximation for the half-levels.
        A[i, i - 1] = Sp[i - 1] / dp1**2
        A[i, i] = -(Sp[i - 1] / dp1**2 + Sp[i] / dp2**2)
        A[i, i + 1] = Sp[i] / dp2**2

    # Boundary conditions (implemented by modifying the first and last rows of A)
    if bc == "neumann":
        # Neumann: dψ/dp = 0 at boundaries (zero flux)
        # Top boundary (i=0): Use a first-order difference approximation: ψ_1 - ψ_0 = 0
        A[0, 0] = -1.0
        A[0, 1] = 1.0
        # Bottom boundary (i=Np-1): Use a first-order difference approximation: ψ_{N-1} - ψ_{N-2} = 0
        A[-1, -1] = -1.0
        A[-1, -2] = 1.0
    elif bc == "dirichlet":
        # Dirichlet: ψ = 0 at boundaries (zero value)
        # Top boundary (i=0): ψ_0 = 0 -> A[0, 0] = 1 (clear other terms)
        A[0, 0] = 1.0
        A[0, 1:] = 0.0 
        # Bottom boundary (i=Np-1): ψ_{N-1} = 0 -> A[-1, -1] = 1 (clear other terms)
        A[-1, -1] = 1.0
        A[-1, :-1] = 0.0 
    else:
        raise ValueError("bc must be 'dirichlet' or 'neumann'")

    # Solve the generalized eigenvalue problem: A * Psi = lam * Psi
    # We solve for -A*Psi = lambda*Psi (A approximates the negative of the operator)
    lam, Psi = eigh(-A) 
    idx = np.argsort(lam) # Sort by ascending eigenvalue (modes are ordered by vertical scale)
    lam, Psi = lam[idx], Psi[:, idx]

    # Retain only the first jmax modes (the physically most relevant low-order modes)
    Psi = Psi[:, :jmax]
    lam = lam[:jmax]

    # Normalize: Enforce the orthonormality condition: ∫ Psi_j^2 * Wp dp = 1
    norm = np.sqrt(np.sum(Wp[:, None] * Psi**2, axis=0))
    Psi /= norm

    return Psi, lam, Wp

def vertical_modes_physical_strict(
    p, T, jmax=4, R=287.0, g=9.81, kappa=0.286, bc="neumann"
):
    """
    Compute physically consistent vertical structure modes L_n(p)
    by solving the temperature-dependent Sturm–Liouville eigenproblem
    derived from the Gill-type vertical structure equation:
    
        (RT/g) d²L/dz² + (R/g dT/dz - 1) dL/dz + (1/h)(R/g dT/dz + κ)L = 0.

    The equation is rewritten in pressure coordinates (p), where z-derivatives
    are replaced using dp/dz = -ρg = -pg/(RT).  This leads to a variable-coefficient
    differential operator that depends on T(p) and dT/dp.

    Parameters
    ----------
    p : (N,) array_like
        Pressure levels [Pa or hPa]. Must be *monotonically increasing downward*.
        If you use hPa, the returned equivalent depths will simply scale accordingly.
    T : (N,) array_like
        Background temperature profile [K] at each pressure level.
    jmax : int, optional
        Number of vertical eigenmodes to retain (default = 4).
    R : float, optional
        Gas constant for dry air [J/kg/K].
    g : float, optional
        Gravitational acceleration [m/s²].
    kappa : float, optional
        Ratio R/c_p (≈ 0.286 for dry air).
    bc : {"neumann","dirichlet"}, optional
        Boundary condition at top and bottom:
        - "neumann": dL/dp = 0  (no vertical flux)
        - "dirichlet": L = 0     (free surface)

    Returns
    -------
    L : (N, jmax) ndarray
        Normalized vertical eigenfunctions L_n(p).
        Columns correspond to successive barotropic/baroclinic modes.
    h : (jmax,) ndarray
        Equivalent depths [m] for each mode (1 / eigenvalue).
    Wp : (N,) ndarray
        Quadrature weights for pressure integration (used for orthonormalization).

    Notes
    -----
    - The problem is cast as a generalized matrix eigenproblem:
        M @ L = λ * W @ L,
      where M is the discretized vertical operator and W is the weighting matrix.
    - Eigenvalues λ = 1/h_n are proportional to inverse equivalent depths.
    - For realistic tropospheric T(p), λ₀ ≪ λ₁ ≪ λ₂.
    """

    # ---------------------------------------------------------------------
    # 1. Ensure numpy arrays and basic shape
    # ---------------------------------------------------------------------
    p = np.asarray(p, dtype=float)
    T = np.asarray(T, dtype=float)
    N = len(p)
    if N < 3:
        raise ValueError("Need at least 3 pressure levels to compute derivatives.")

    # ---------------------------------------------------------------------
    # 2. Compute temperature gradient wrt pressure
    # ---------------------------------------------------------------------
    # This gives dT/dp at each level using central differences
    dTdp = np.gradient(T, p)

    # ---------------------------------------------------------------------
    # 3. Define coefficient functions A(p), B(p), and C(p)
    # ---------------------------------------------------------------------
    # From the pressure-coordinate form of the equation:
    #   p² L_pp + p(2 + p/T dT/dp) L_p + (R p² / g h)(κ - p/T dT/dp)L = 0
    #
    # Rearranged to standard Sturm–Liouville form:
    #   -d/dp(A dL/dp) - B dL/dp = (C / h) L
    # so that M L = λ W L,  with λ = 1/h
    #
    A = p**2                            # coefficient for second derivative term
    B = p * (dTdp / T)                  # coefficient for first derivative term
    C = (R * p**2 / g) * (kappa - (p / T) * dTdp)  # weighting for RHS

    # ---------------------------------------------------------------------
    # 4. Build finite-difference derivative operators
    # ---------------------------------------------------------------------
    # D1 approximates dL/dp, D2 approximates d²L/dp².
    D1 = np.zeros((N, N))
    D2 = np.zeros((N, N))
    for i in range(1, N - 1):
        dp1 = p[i] - p[i - 1]
        dp2 = p[i + 1] - p[i]
        # First derivative (central difference, antisymmetric stencil)
        D1[i, i - 1] = -1.0 / dp1
        D1[i, i + 1] =  1.0 / dp2
        # Second derivative (central difference, symmetric stencil)
        D2[i, i - 1] =  1.0 / dp1**2
        D2[i, i]   = -1.0 / dp1**2 - 1.0 / dp2**2
        D2[i, i + 1] =  1.0 / dp2**2

    # ---------------------------------------------------------------------
    # 5. Apply boundary conditions
    # ---------------------------------------------------------------------
    if bc == "neumann":
        # Zero gradient at top and bottom: dL/dp = 0
        # Implemented as (L₁ - L₀) = 0 and (L_N - L_{N-1}) = 0
        D1[0, 0], D1[0, 1] = -1, 1
        D1[-1, -2], D1[-1, -1] = -1, 1
    elif bc == "dirichlet":
        # Fixed value (L=0) at top and bottom
        D2[0, :] = 0; D2[0, 0] = 1
        D2[-1, :] = 0; D2[-1, -1] = 1
    else:
        raise ValueError("bc must be 'neumann' or 'dirichlet'")

    # ---------------------------------------------------------------------
    # 6. Assemble the vertical operator M and weighting matrix W
    # ---------------------------------------------------------------------
    # M = -A * D2 - B * D1  (matrix multiplication implied)
    M = -np.diag(A) @ D2 - np.diag(B) @ D1

    # Weighting matrix (acts like RHS in the generalized eigenproblem)
    W = np.diag(C)

    # ---------------------------------------------------------------------
    # 7. Solve generalized eigenvalue problem M L = λ W L
    # ---------------------------------------------------------------------
    lam, L = eigh(M, W)

    # ---------------------------------------------------------------------
    # 8. Sort and retain leading jmax physically relevant modes
    # ---------------------------------------------------------------------
    idx = np.argsort(lam)
    lam, L = lam[idx], L[:, idx]
    h = 1.0 / lam[:jmax]   # equivalent depths
    L = L[:, :jmax]        # eigenfunctions

    # ---------------------------------------------------------------------
    # 9. Normalize eigenfunctions with respect to ∫ L² dp = 1
    # ---------------------------------------------------------------------
    Wp = np.abs(np.gradient(p))
    norm = np.sqrt(np.sum(Wp[:, None] * L**2, axis=0))
    L /= norm

    # ---------------------------------------------------------------------
    # 10. Return eigenfunctions, equivalent depths, and weights
    # ---------------------------------------------------------------------
    return L, h, Wp

# ---------------------------------------------------------------------
# Projection (Q(r,p) -> coefficients C[n,j])
# ---------------------------------------------------------------------
def project_coefficients(Q, r, p, Phi, Wr, Psi, Wp):
    """
    Compute spectral coefficients C[n,j] for Q(r,p)
    using orthonormal radial (Phi) and vertical (Psi) bases.

    Q must have shape (Nr, Np)
    The projection integral is: C_nj = ∫∫ Q(r,p) * Phi_n(r) * Psi_j(p) * r dr dp
    """
    Q = np.asarray(Q)

    # Autodetect and transpose if shape is (Np, Nr)
    if Q.shape[0] == len(p):
        Q = Q.T # Transpose to (Nr, Np)

    # Step 1: Radial weighting. Wr[:, None] * Q: Apply radial weighting (r dr) to the field Q
    Q_weighted_r = Wr[:, None] * Q # (Nr, Np)

    # Step 2: Integrate over p (vertical dimension)
    # This matrix multiplication is: Σ_p (Q * r dr) * (Psi * dp)
    # The result is (Nr, jmax) matrix: ∫ Q(r,p) * Psi_j(p) dp * r dr
    Q_integrated_p = Q_weighted_r @ (Psi * Wp[:, None])

    # Step 3: Integrate over r (radial dimension)
    # Matrix multiplication: (nmax, Nr) @ (Nr, jmax) -> (nmax, jmax)
    # This completes the double integral.
    C = Phi.T @ Q_integrated_p
    return C

# ---------------------------------------------------------------------
# Reconstruction (coefficients C[n,j] -> Q(r,p))
# ---------------------------------------------------------------------
def reconstruct_field(C, Phi, Psi):
    """
    Reconstruct Q(r,p) from coefficients C[n,j].
    Reconstruction: Q_hat(r,p) = sum_n sum_j C_nj * Phi_n(r) * Psi_j(p)
    """
    # Phi @ C: (Nr, nmax) @ (nmax, jmax) -> (Nr, jmax) - reconstructs the vertical slices
    # (Phi @ C) @ Psi.T: (Nr, jmax) @ (jmax, Np) -> (Nr, Np) - reconstructs the full field
    return Phi @ C @ Psi.T
