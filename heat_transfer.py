"""
Heat Transfer Module 

This module implements heat transfer calculations including:
- Fin efficiency and effectiveness
- Convection coefficient calculations
- Total heat dissipation
- Base temperature computation
"""

import numpy as np
from typing import Tuple, Dict
from config import config


def calculate_fin_efficiency(H: float, t: float, h: float, k: float) -> float:
    """
    Calculate fin efficiency using hyperbolic tangent solution.
    
    For a rectangular fin with adiabatic tip:
    η_fin = tanh(m·L) / (m·L)
    where m = sqrt(h·P / (k·A_c))
    
    Args:
        H: Fin height [m]
        t: Fin thickness [m]
        h: Convection coefficient [W/m²·K]
        k: Thermal conductivity of fin material [W/m·K]
        
    Returns:
        eta_fin: Fin efficiency [-]
    
    Theory:
        The fin efficiency represents the ratio of actual heat transfer
        to ideal heat transfer if the entire fin were at base temperature.
        Lower efficiency indicates larger temperature drop along fin.
    """
    # Fin perimeter (assuming rectangular cross-section)
    # For thin fins, P ≈ 2 × (width + thickness) ≈ 2 × width
    # Since width >> thickness, P ≈ 2 × base_width
    base_width = config.operating.base_width
    P = 2 * (base_width + t)
    
    # Cross-sectional area of fin
    A_c = base_width * t
    
    # Fin parameter m [1/m]
    m = np.sqrt(h * P / (k * A_c))
    
    # Dimensionless parameter
    mL = m * H
    
    # Fin efficiency
    # Handle numerical issues for very small mL
    if mL < 0.001:
        eta_fin = 1.0  # Very short fin, nearly 100% efficient
    else:
        eta_fin = np.tanh(mL) / mL
    
    return eta_fin


def calculate_fin_effectiveness(eta_fin: float, A_fin: float, 
                                A_base: float, h: float) -> float:
    """
    Calculate fin effectiveness.
    
    ε = (η_fin·A_fin·h) / (A_base·h) = η_fin·(A_fin/A_base)
    
    Args:
        eta_fin: Fin efficiency [-]
        A_fin: Surface area of one fin [m²]
        A_base: Base area occupied by one fin [m²]
        h: Convection coefficient [W/m²·K]
        
    Returns:
        epsilon: Fin effectiveness [-]
        
    Theory:
        Effectiveness > 1 means fin enhances heat transfer.
        Typically want ε > 2 for fins to be worthwhile.
    """
    epsilon = eta_fin * (A_fin / A_base)
    return epsilon


def calculate_fin_surface_area(H: float, t: float, W: float) -> float:
    """
    Calculate surface area of one rectangular fin.
    
    Args:
        H: Fin height [m]
        t: Fin thickness [m]
        W: Fin width (base width) [m]
        
    Returns:
        A_fin: Surface area of fin [m²]
    """
    # Two sides (height × width) + tip (thickness × width)
    # Neglecting tip area for thin fins: A ≈ 2·H·W
    A_fin = 2 * H * W + t * W  # Include tip for accuracy
    return A_fin


def calculate_total_heat_transfer(N: int, H: float, t: float, s: float,
                                  h: float, T_base: float, T_inf: float,
                                  k: float) -> float:
    """
    Calculate total heat dissipation from finned heat sink.
    
    Q_total = Q_fins + Q_base_exposed
    Q_fins = N · η_fin · h · A_fin · (T_base - T_inf)
    Q_base = h · A_base_exposed · (T_base - T_inf)
    
    Args:
        N: Number of fins [-]
        H: Fin height [m]
        t: Fin thickness [m]
        s: Fin spacing [m]
        h: Convection coefficient [W/m²·K]
        T_base: Base temperature [K or °C]
        T_inf: Ambient temperature [K or °C]
        k: Thermal conductivity of fin material [W/m·K]
        
    Returns:
        Q_total: Total heat dissipation [W]
    """
    W = config.operating.base_width
    L = config.operating.base_length
    
    # Fin efficiency
    eta_fin = calculate_fin_efficiency(H, t, h, k)
    
    # Surface area of one fin
    A_fin = calculate_fin_surface_area(H, t, W)
    
    # Heat transfer from fins
    Q_fins = N * eta_fin * h * A_fin * (T_base - T_inf)
    
    # Exposed base area (area not covered by fin bases)
    A_base_total = W * L
    A_fins_base = N * t * W
    A_base_exposed = A_base_total - A_fins_base
    
    # Heat transfer from exposed base
    if A_base_exposed > 0:
        Q_base = h * A_base_exposed * (T_base - T_inf)
    else:
        Q_base = 0.0
    
    # Total heat transfer
    Q_total = Q_fins + Q_base
    
    return Q_total


def solve_base_temperature(N: int, H: float, t: float, s: float, v: float,
                          Q_load: float, T_inf: float, 
                          tol: float = 0.1, max_iter: int = 100) -> Tuple[float, Dict]:
    """
    Solve for base temperature using iterative method.
    
    Since h depends on T (through properties) and T_base is unknown,
    we need to iterate:
    1. Guess T_base
    2. Calculate average temperature for properties
    3. Calculate h using convection correlations
    4. Calculate Q_total with this h and T_base
    5. Check if Q_total ≈ Q_load
    6. Update T_base and repeat
    
    Args:
        N: Number of fins [-]
        H: Fin height [m]
        t: Fin thickness [m]
        s: Fin spacing [m]
        v: Air velocity [m/s]
        Q_load: Required heat dissipation [W]
        T_inf: Ambient temperature [°C]
        tol: Convergence tolerance [°C]
        max_iter: Maximum iterations
        
    Returns:
        T_base: Base temperature [°C]
        info: Dictionary with convergence information
    """
    from fluid_dynamics import calculate_convection_coefficient
    
    k = config.materials.aluminium_k
    
    # Initial guess for T_base
    T_base = T_inf + 50.0  # Start with 50°C above ambient
    
    converged = False
    iteration = 0
    residual = 0.0
    
    for iteration in range(max_iter):
        # Calculate convection coefficient with current T_base guess
        h = calculate_convection_coefficient(v, s, H, T_base, T_inf)
        
        # Calculate heat dissipation with this h and T_base
        Q_calc = calculate_total_heat_transfer(N, H, t, s, h, T_base, T_inf, k)
        
        # Check convergence
        residual = Q_calc - Q_load
        
        if abs(residual) < tol:
            converged = True
            break
        
        # Update T_base using Newton-like iteration
        # If Q_calc > Q_load, heat sink is too effective, reduce T_base
        # If Q_calc < Q_load, heat sink is insufficient, increase T_base
        
        # Numerical derivative: dQ/dT
        dT = 0.1
        Q_plus = calculate_total_heat_transfer(N, H, t, s, h, T_base + dT, T_inf, k)
        dQ_dT = (Q_plus - Q_calc) / dT
        
        # Newton update with damping
        if abs(dQ_dT) > 1e-6:
            T_base_new = T_base - 0.5 * residual / dQ_dT
        else:
            # Fallback to bisection-like update
            T_base_new = T_base - 0.1 * np.sign(residual) * abs(residual) / Q_load * 10
        
        # Ensure reasonable bounds
        T_base = np.clip(T_base_new, T_inf + 1, T_inf + 100)
    
    info = {
        'converged': converged,
        'iterations': iteration + 1,
        'final_residual': abs(residual),
        'Q_calculated': Q_calc if iteration < max_iter else 0,
        'Q_required': Q_load,
        'h': h if iteration < max_iter else 0
    }
    
    return T_base, info


def calculate_thermal_resistance(T_base: float, T_inf: float, Q: float) -> float:
    """
    Calculate thermal resistance of heat sink.
    
    R_th = (T_base - T_inf) / Q
    
    Args:
        T_base: Base temperature [°C]
        T_inf: Ambient temperature [°C]
        Q: Heat dissipation [W]
        
    Returns:
        R_th: Thermal resistance [K/W or °C/W]
    """
    if Q < 1e-6:
        return np.inf
    
    R_th = (T_base - T_inf) / Q
    return R_th


def calculate_fin_temperature_profile(H: float, t: float, h: float, k: float,
                                     T_base: float, T_inf: float, 
                                     n_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate temperature distribution along fin height.
    
    For visualization purposes - shows how temperature drops along fin.
    
    Args:
        H: Fin height [m]
        t: Fin thickness [m]
        h: Convection coefficient [W/m²·K]
        k: Thermal conductivity [W/m·K]
        T_base: Base temperature [°C]
        T_inf: Ambient temperature [°C]
        n_points: Number of points to calculate
        
    Returns:
        x: Position along fin [m]
        T: Temperature at each position [°C]
    """
    base_width = config.operating.base_width
    P = 2 * (base_width + t)
    A_c = base_width * t
    
    # Fin parameter
    m = np.sqrt(h * P / (k * A_c))
    
    # Position along fin
    x = np.linspace(0, H, n_points)
    
    # Temperature distribution (adiabatic tip)
    # T(x) = T_inf + (T_base - T_inf) * cosh(m*(H-x)) / cosh(m*H)
    T = T_inf + (T_base - T_inf) * np.cosh(m * (H - x)) / np.cosh(m * H)
    
    return x, T


def validate_heat_transfer_calculation():
    """
    Validate heat transfer calculations with known analytical solutions.
    """
    print("\n" + "="*70)
    print("VALIDATING HEAT TRANSFER CALCULATIONS")
    print("="*70)
    
    # Test case 1: Fin efficiency
    H = 0.05  # 50 mm
    t = 0.002  # 2 mm
    h = 50.0  # W/m²·K
    k = 205.0  # Aluminum
    
    eta_fin = calculate_fin_efficiency(H, t, h, k)
    print(f"\nTest 1: Fin Efficiency")
    print(f"  Fin height: {H*1000:.1f} mm")
    print(f"  Fin thickness: {t*1000:.1f} mm")
    print(f"  h: {h:.1f} W/m²·K")
    print(f"  Efficiency: {eta_fin:.4f}")
    print(f"  Expected: ~0.85-0.95 (reasonable for these parameters)")
    
    # Test case 2: Base temperature solver
    print(f"\nTest 2: Base Temperature Solver")
    N = 20
    H = 0.04
    t = 0.002
    s = 0.005
    v = 3.0
    Q_load = 100.0
    T_inf = 25.0
    
    T_base, info = solve_base_temperature(N, H, t, s, v, Q_load, T_inf)
    print(f"  Design: N={N}, H={H*1000:.0f}mm, t={t*1000:.0f}mm, s={s*1000:.0f}mm, v={v:.0f}m/s")
    print(f"  Heat load: {Q_load:.0f} W")
    print(f"  Base temperature: {T_base:.2f} °C")
    print(f"  Converged: {info['converged']}")
    print(f"  Iterations: {info['iterations']}")
    print(f"  Final residual: {info['final_residual']:.4f} W")
    
    R_th = calculate_thermal_resistance(T_base, T_inf, Q_load)
    print(f"  Thermal resistance: {R_th:.4f} K/W")
    
    # Test case 3: Temperature profile
    print(f"\nTest 3: Fin Temperature Profile")
    x, T = calculate_fin_temperature_profile(H, t, h, k, T_base, T_inf, n_points=5)
    print(f"  Position [mm]:     Temperature [°C]:")
    for i in range(len(x)):
        print(f"    {x[i]*1000:6.1f}           {T[i]:6.2f}")
    
    print("="*70)


if __name__ == "__main__":
    # Run validation
    validate_heat_transfer_calculation()