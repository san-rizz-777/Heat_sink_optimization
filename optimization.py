"""
Optimization Module for Heat Sink Design
Course: Transport Phenomena
Author: Sanskar Gunde (MM24B005)
Date: November 2025

This module implements the Differential Evolution optimization
algorithm for heat sink design optimization.
"""

import numpy as np
from scipy.optimize import differential_evolution
from typing import Tuple, Dict, Callable
import time

from config import config
from heat_transfer import solve_base_temperature
from fluid_dynamics import calculate_pressure_drop, calculate_fan_power
from performance import calculate_material_cost, calculate_material_mass


def objective_function(x: np.ndarray, verbose: bool = False) -> float:
    """
    Multi-objective function for heat sink optimization.
    
    Minimize: J = w1*(T_norm) + w2*(P_norm) + w3*(Cost_norm)
    
    Where each term is normalized to [0, 1] range for balanced optimization.
    
    Args:
        x: Design vector [N, H, t, s, v]
        verbose: Print detailed information
        
    Returns:
        J: Objective function value (lower is better)
    """
    # Extract design variables
    N = int(round(x[0]))  # Number of fins (integer)
    H = x[1]  # Fin height [m]
    t = x[2]  # Fin thickness [m]
    s = x[3]  # Fin spacing [m]
    v = x[4]  # Air velocity [m/s]
    
    # Validate design constraints
    is_valid, msg = config.validate_design(x)
    if not is_valid:
        if verbose:
            print(f"  Invalid design: {msg}")
        return 1e6  # Large penalty for invalid designs
    
    # Operating conditions
    Q_load = config.operating.Q_cpu
    T_inf = config.operating.T_ambient
    T_target = config.operating.T_target
    T_max = config.operating.T_max
    
    try:
        # Solve for base temperature
        T_base, info = solve_base_temperature(N, H, t, s, v, Q_load, T_inf,
                                             tol=0.5, max_iter=50)
        
        if not info['converged']:
            if verbose:
                print(f"  Temperature solver did not converge")
            return 1e6
        
        # Calculate pressure drop
        T_film = (T_base + T_inf) / 2.0
        delta_P = calculate_pressure_drop(N, s, H, v, T_film)
        
        # Calculate fan power
        P_fan = calculate_fan_power(delta_P, v, N, s)
        
        # Calculate material cost
        mass = calculate_material_mass(N, H, t)
        cost = calculate_material_cost(mass)
        
        # Check thermal constraint - HARD penalty if exceeded
        if T_base > T_max:
            penalty = 1000 * (T_base - T_max)
            if verbose:
                print(f"  WARNING: T_base={T_base:.1f}°C exceeds T_max={T_max:.1f}°C")
            return penalty
        
        # NORMALIZED objective function for balanced optimization
        # This prevents any single term from dominating
        
        # Normalization ranges (typical expected values)
        T_range = T_max - T_target  # e.g., 85-70 = 15°C
        P_fan_max = 10.0  # Max expected fan power [W]
        cost_max = 1.0  # Max expected cost [$]
        
        # Temperature term: normalized to [0, 1]
        # 0 = at or below target, 1 = at maximum allowed
        if T_base <= T_target:
            temp_normalized = 0.0  # Perfect!
        else:
            temp_normalized = (T_base - T_target) / T_range
            temp_normalized = min(temp_normalized, 1.0)
        
        # Power term: normalized to [0, 1]
        power_normalized = P_fan / P_fan_max
        power_normalized = min(power_normalized, 1.0)
        
        # Cost term: normalized to [0, 1]
        cost_normalized = cost / cost_max
        cost_normalized = min(cost_normalized, 1.0)
        
        # Weighted objective function
        w1 = config.optimization.weight_temperature
        w2 = config.optimization.weight_power
        w3 = config.optimization.weight_cost
        
        # Normalize weights to sum to 1
        w_total = w1 + w2 + w3
        w1_norm = w1 / w_total
        w2_norm = w2 / w_total
        w3_norm = w3 / w_total
        
        J = w1_norm * temp_normalized + w2_norm * power_normalized + w3_norm * cost_normalized
        
        # J is now in range [0, 1] where:
        # 0 = perfect design (meets all objectives)
        # 1 = worst acceptable design
        
        if verbose:
            print(f"\n  Design: N={N}, H={H*1000:.1f}mm, t={t*1000:.2f}mm, "
                  f"s={s*1000:.1f}mm, v={v:.1f}m/s")
            print(f"  T_base: {T_base:.2f}°C, P_fan: {P_fan:.3f}W, "
                  f"Cost: ${cost:.2f}, Mass: {mass*1000:.1f}g")
            print(f"  Objective: {J:.4f} (normalized)")
            print(f"    Temperature: {temp_normalized:.3f} ({T_base:.1f}°C vs target {T_target:.1f}°C)")
            print(f"    Power: {power_normalized:.3f} ({P_fan:.3f}W)")
            print(f"    Cost: {cost_normalized:.3f} (${cost:.2f})")
        
        return J
    
    except Exception as e:
        if verbose:
            print(f"  Error in objective function: {e}")
            import traceback
            traceback.print_exc()
        return 1e6  # Return large value on error


def constraint_spacing(x: np.ndarray) -> float:
    """
    Constraint: s >= 2*t (manufacturability).
    
    Returns positive value if constraint is satisfied.
    """
    t = x[2]
    s = x[3]
    return s - 2 * t


def constraint_max_width(x: np.ndarray) -> float:
    """
    Constraint: N*(t+s) <= W_max.
    
    Returns positive value if constraint is satisfied.
    """
    N = x[0]
    t = x[2]
    s = x[3]
    W_max = config.constraints.max_total_width
    return W_max - N * (t + s)


def constraint_temperature(x: np.ndarray) -> float:
    """
    Constraint: T_base <= T_max.
    
    Returns positive value if constraint is satisfied.
    """
    N = int(round(x[0]))
    H = x[1]
    t = x[2]
    s = x[3]
    v = x[4]
    
    Q_load = config.operating.Q_cpu
    T_inf = config.operating.T_ambient
    T_max = config.operating.T_max
    
    try:
        T_base, info = solve_base_temperature(N, H, t, s, v, Q_load, T_inf,
                                             tol=1.0, max_iter=30)
        return T_max - T_base
    except:
        return -1.0  # Constraint violated


def optimize_heat_sink(verbose: bool = True) -> Tuple[np.ndarray, float, Dict]:
    """
    Optimize heat sink design using Differential Evolution.
    
    Args:
        verbose: Print optimization progress
        
    Returns:
        x_opt: Optimal design vector [N, H, t, s, v]
        J_opt: Optimal objective value
        result: Full optimization result dictionary
    """
    if verbose:
        print("\n" + "="*70)
        print("HEAT SINK OPTIMIZATION USING DIFFERENTIAL EVOLUTION")
        print("="*70)
        print("\nOptimization Parameters:")
        print(f"  Algorithm: Differential Evolution")
        print(f"  Strategy: {config.optimization.strategy}")
        print(f"  Population size: {config.optimization.popsize} × dimensions")
        print(f"  Max iterations: {config.optimization.maxiter}")
        print(f"  Mutation: {config.optimization.mutation}")
        print(f"  Recombination: {config.optimization.recombination}")
    
    # Design variable bounds
    bounds = config.bounds.get_bounds_array()
    
    # Callback function for progress reporting
    iteration = [0]
    best_values = []
    
    def callback(xk, convergence):
        iteration[0] += 1
        J = objective_function(xk)
        best_values.append(J)
        
        if verbose and iteration[0] % 20 == 0:
            N = int(round(xk[0]))
            print(f"  Iteration {iteration[0]}: "
                  f"J={J:.4f}, N={N}, "
                  f"H={xk[1]*1000:.1f}mm, v={xk[4]:.1f}m/s")
        
        return False  # Continue optimization
    
    # Run optimization
    if verbose:
        print("\nStarting optimization...")
        print("-" * 70)
    
    start_time = time.time()
    
    result = differential_evolution(
        objective_function,
        bounds=bounds,
        strategy=config.optimization.strategy,
        maxiter=config.optimization.maxiter,
        popsize=config.optimization.popsize,
        tol=config.optimization.tol,
        mutation=config.optimization.mutation,
        recombination=config.optimization.recombination,
        seed=config.optimization.seed,
        callback=callback,
        polish=config.optimization.polish,
        workers=config.optimization.workers,
        updating=config.optimization.updating,
        atol=config.optimization.atol,
        disp=False
    )
    
    elapsed_time = time.time() - start_time
    
    x_opt = result.x
    J_opt = result.fun
    
    if verbose:
        print("-" * 70)
        print(f"\nOptimization completed in {elapsed_time:.2f} seconds")
        print(f"Total function evaluations: {result.nfev}")
        print(f"Success: {result.success}")
        if not result.success:
            print(f"Message: {result.message}")
    
    # Add convergence history to result
    result_dict = {
        'x_opt': x_opt,
        'J_opt': J_opt,
        'success': result.success,
        'message': result.message,
        'nfev': result.nfev,
        'nit': result.nit,
        'elapsed_time': elapsed_time,
        'convergence_history': best_values
    }
    
    return x_opt, J_opt, result_dict


def evaluate_design(x: np.ndarray, verbose: bool = True) -> Dict:
    """
    Evaluate a design and return all performance metrics.
    
    Args:
        x: Design vector [N, H, t, s, v]
        verbose: Print detailed information
        
    Returns:
        metrics: Dictionary of performance metrics
    """
    from performance import calculate_all_metrics
    
    N = int(round(x[0]))
    H = x[1]
    t = x[2]
    s = x[3]
    v = x[4]
    
    Q_load = config.operating.Q_cpu
    T_inf = config.operating.T_ambient
    
    # Solve for base temperature
    T_base, info = solve_base_temperature(N, H, t, s, v, Q_load, T_inf)
    
    # Calculate all metrics
    metrics = calculate_all_metrics(x, T_base, info)
    
    if verbose:
        print("\n" + "="*70)
        print("DESIGN EVALUATION")
        print("="*70)
        print(f"\nDesign Parameters:")
        print(f"  Number of fins (N): {N}")
        print(f"  Fin height (H): {H*1000:.2f} mm")
        print(f"  Fin thickness (t): {t*1000:.2f} mm")
        print(f"  Fin spacing (s): {s*1000:.2f} mm")
        print(f"  Air velocity (v): {v:.2f} m/s")
        
        print(f"\nThermal Performance:")
        print(f"  Base temperature: {T_base:.2f} °C")
        print(f"  Temperature rise: {T_base - T_inf:.2f} °C")
        print(f"  Thermal resistance: {metrics['thermal_resistance']:.4f} K/W")
        print(f"  Convection coefficient: {metrics['convection_coeff']:.2f} W/m²·K")
        
        print(f"\nFluid Dynamics:")
        print(f"  Reynolds number: {metrics['reynolds_number']:.0f}")
        print(f"  Flow regime: {metrics['flow_regime']}")
        print(f"  Pressure drop: {metrics['pressure_drop']:.2f} Pa")
        print(f"  Fan power: {metrics['fan_power']:.3f} W")
        
        print(f"\nMaterial & Cost:")
        print(f"  Total mass: {metrics['total_mass']*1000:.2f} g")
        print(f"  Material cost: ${metrics['material_cost']:.2f}")
        
        print(f"\nFigures of Merit:")
        print(f"  Specific cooling: {metrics['specific_cooling']:.2f} W/kg")
        print(f"  Cost per watt: ${metrics['cost_per_watt']:.4f}/W")
        print(f"  Total power: {metrics['total_system_power']:.2f} W")
        print("="*70)
    
    return metrics


if __name__ == "__main__":
    # Test optimization
    print("Testing objective function...")
    
    # Test with baseline design
    x_baseline = config.get_baseline_design()
    print("\nEvaluating baseline design:")
    J_baseline = objective_function(x_baseline, verbose=True)
    print(f"\nBaseline objective value: {J_baseline:.4f}")
    
    # Run optimization (with reduced iterations for testing)
    print("\n\nRunning optimization (limited iterations for testing)...")
    config.optimization.maxiter = 50  # Reduce for quick test
    x_opt, J_opt, result = optimize_heat_sink(verbose=True)
    
    print("\n\nOptimal design:")
    metrics_opt = evaluate_design(x_opt, verbose=True)
    
    print(f"\n\nImprovement:")
    print(f"  Objective: {J_baseline:.4f} → {J_opt:.4f} "
          f"({(1-J_opt/J_baseline)*100:.1f}% better)")