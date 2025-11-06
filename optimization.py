"""
Optimization Module for Heat Sink Design
Course: Transport Phenomena
Author: Sanskar Gunde (MM24B005)
Date: November 2025

This module implements the Differential Evolution optimization
algorithm for heat sink design optimization with realistic physics constraints.
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
    Multi-objective function for heat sink optimization with penalty method.
    
    Minimize: J = w1*(T_norm) + w2*(P_norm) + w3*(Cost_norm) + penalties
    
    Where each term is normalized to [0, 1] range for balanced optimization.
    Penalties enforce physical constraints that are critical for realistic designs.
    
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
    
    # Operating conditions
    Q_load = config.operating.Q_cpu
    T_inf = config.operating.T_ambient
    T_target = config.operating.T_target
    T_max = config.operating.T_max
    
    # ============================================================
    # PENALTY METHOD FOR CONSTRAINTS
    # ============================================================
    penalty = 0.0
    penalty_weight = 1000.0  # Large weight for hard constraints
    
    # 1. Minimum spacing constraint (manufacturability and dust)
    s_min_m = 0.003  # 3mm minimum
    if s < s_min_m:
        penalty += penalty_weight * (s_min_m - s)**2
    
    # 2. Aspect ratio constraint (structural stability)
    aspect_ratio = H / t
    aspect_max = 30.0
    if aspect_ratio > aspect_max:
        penalty += penalty_weight * (aspect_ratio - aspect_max)**2
    
    # 3. Maximum velocity constraint (noise limit)
    v_max = 8.0  # m/s
    if v > v_max:
        penalty += penalty_weight * (v - v_max)**2
    
    # 4. Minimum velocity constraint (effectiveness)
    v_min = 2.0  # m/s
    if v < v_min:
        penalty += penalty_weight * (v_min - v)**2
    
    # 5. Total width constraint (physical fit)
    total_width = N * t + (N - 1) * s  # [m]
    width_max = config.constraints.max_total_width
    if total_width > width_max:
        penalty += penalty_weight * (total_width - width_max)**2
    
    # 6. Minimum fin thickness (structural integrity)
    t_min = 0.001  # 1mm
    if t < t_min:
        penalty += penalty_weight * (t_min - t)**2
    
    # 7. Minimum fin height (effectiveness)
    H_min = 0.015  # 15mm
    if H < H_min:
        penalty += penalty_weight * (H_min - H)**2
    
    # 8. Spacing-to-thickness ratio (manufacturability)
    if s < 2.0 * t:
        penalty += penalty_weight * (2.0 * t - s)**2
    
    # If basic geometry constraints violated, return penalty immediately
    if penalty > 0:
        if verbose:
            print(f"  Constraint violation penalty: {penalty:.2f}")
        return penalty
    
    try:
        # ============================================================
        # SOLVE FOR THERMAL PERFORMANCE
        # ============================================================
        T_base, info = solve_base_temperature(N, H, t, s, v, Q_load, T_inf,
                                             tol=0.5, max_iter=50)
        
        if not info['converged']:
            if verbose:
                print(f"  Temperature solver did not converge")
            return 1e6
        
        # Calculate fluid properties at film temperature
        T_film = (T_base + T_inf) / 2.0
        
        # Calculate pressure drop
        delta_P = calculate_pressure_drop(N, s, H, v, T_film)
        
        # Calculate fan power (CRITICAL: must use realistic model)
        P_fan = calculate_fan_power(delta_P, v, N, s)
        
        # Calculate material cost
        mass = calculate_material_mass(N, H, t)
        cost = calculate_material_cost(mass)
        
        # ============================================================
        # PHYSICS-BASED VALIDATION CHECKS
        # ============================================================
        
        # Check 1: Reynolds number sanity
        rho = 1.184  # kg/m³ at 25°C
        mu = 1.849e-5  # Pa·s
        Dh = 2 * s * H / (s + H)  # Hydraulic diameter [m]
        Re = rho * v * Dh / mu
        
        if Re < 100:  # Too low, poor heat transfer
            penalty += penalty_weight * (100 - Re)**2
        elif Re > 50000:  # Unrealistically high
            penalty += penalty_weight * (Re - 50000)**2
        
        # Check 2: Fan power scaling (must follow P ∝ v³ approximately)
        # Power ratio should be in realistic range
        power_per_v_cubed = P_fan / (v**3)
        if power_per_v_cubed < 0.001:  # Physics violated
            penalty += penalty_weight * 1000
        elif power_per_v_cubed > 10.0:  # Unrealistically high
            penalty += penalty_weight * (power_per_v_cubed - 10.0)**2
        
        # Check 3: Pressure drop reasonableness
        if delta_P < 0.1:  # Too low, likely calculation error
            penalty += penalty_weight * 100
        elif delta_P > 1000:  # Unrealistically high (>1kPa)
            penalty += penalty_weight * (delta_P - 1000)**2
        
        # Check 4: Temperature reasonableness
        if T_base < T_inf + 5:  # Less than 5°C rise is unrealistic
            penalty += penalty_weight * (T_inf + 5 - T_base)**2
        elif T_base > 200:  # Physically impossible for this application
            penalty += penalty_weight * (T_base - 200)**2
        
        # If validation penalties exist, add them and return
        if penalty > 0:
            if verbose:
                print(f"  Physics validation penalty: {penalty:.2f}")
                print(f"    Re={Re:.0f}, P_fan={P_fan:.3f}W, ΔP={delta_P:.1f}Pa")
            return 1e4 + penalty
        
        # ============================================================
        # HARD CONSTRAINT: Maximum Temperature
        # ============================================================
        if T_base > T_max:
            temp_penalty = 1000 * (T_base - T_max)**2
            if verbose:
                print(f"  CRITICAL: T_base={T_base:.1f}°C exceeds T_max={T_max:.1f}°C")
            return temp_penalty
        
        # ============================================================
        # NORMALIZED MULTI-OBJECTIVE FUNCTION
        # ============================================================
        
        # Normalization ranges (based on expected realistic values)
        T_range = T_max - T_target  # e.g., 85-70 = 15°C
        P_fan_max = 10.0  # Max expected fan power [W] (realistic for desktop)
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
        # 0 = perfect design (meets all objectives optimally)
        # 1 = worst acceptable design (at limits)
        
        if verbose:
            print(f"\n  Design: N={N}, H={H*1000:.1f}mm, t={t*1000:.2f}mm, "
                  f"s={s*1000:.1f}mm, v={v:.1f}m/s")
            print(f"  Performance:")
            print(f"    T_base: {T_base:.2f}°C (target: {T_target}°C, max: {T_max}°C)")
            print(f"    P_fan: {P_fan:.3f}W")
            print(f"    Cost: ${cost:.2f}")
            print(f"    Mass: {mass*1000:.1f}g")
            print(f"  Flow characteristics:")
            print(f"    Re: {Re:.0f}")
            print(f"    ΔP: {delta_P:.1f} Pa")
            print(f"    Regime: {'Laminar' if Re < 2300 else 'Turbulent'}")
            print(f"  Objective: J={J:.4f}")
            print(f"    Temp contribution: {temp_normalized:.3f} (weight: {w1_norm:.2f})")
            print(f"    Power contribution: {power_normalized:.3f} (weight: {w2_norm:.2f})")
            print(f"    Cost contribution: {cost_normalized:.3f} (weight: {w3_norm:.2f})")
        
        return J
    
    except Exception as e:
        if verbose:
            print(f"  Error in objective function: {e}")
            import traceback
            traceback.print_exc()
        return 1e6  # Return large value on error


def optimize_heat_sink(verbose: bool = True) -> Tuple[np.ndarray, float, Dict]:
    """
    Optimize heat sink design using Differential Evolution.
    
    This uses a global optimization algorithm that:
    - Explores the entire design space
    - Handles integer variables (number of fins)
    - Doesn't require gradients
    - Is robust to local minima
    
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
        print(f"\nObjective Weights:")
        print(f"  Temperature: {config.optimization.weight_temperature}")
        print(f"  Power: {config.optimization.weight_power}")
        print(f"  Cost: {config.optimization.weight_cost}")
    
    # Design variable bounds
    bounds = config.bounds.get_bounds_array()
    
    if verbose:
        print(f"\nDesign Variable Bounds:")
        print(f"  N (fins): {bounds[0][0]:.0f} - {bounds[0][1]:.0f}")
        print(f"  H (height): {bounds[1][0]*1000:.1f} - {bounds[1][1]*1000:.1f} mm")
        print(f"  t (thickness): {bounds[2][0]*1000:.2f} - {bounds[2][1]*1000:.2f} mm")
        print(f"  s (spacing): {bounds[3][0]*1000:.2f} - {bounds[3][1]*1000:.2f} mm")
        print(f"  v (velocity): {bounds[4][0]:.1f} - {bounds[4][1]:.1f} m/s")
    
    # Callback function for progress reporting
    iteration = [0]
    best_values = []
    best_designs = []
    
    def callback(xk, convergence):
        """Track optimization progress"""
        iteration[0] += 1
        J = objective_function(xk)
        best_values.append(J)
        best_designs.append(xk.copy())
        
        if verbose and iteration[0] % 20 == 0:
            N = int(round(xk[0]))
            H_mm = xk[1] * 1000
            t_mm = xk[2] * 1000
            s_mm = xk[3] * 1000
            v = xk[4]
            
            print(f"  Iter {iteration[0]:3d}: J={J:8.4f} | "
                  f"N={N:2d}, H={H_mm:5.1f}mm, t={t_mm:4.1f}mm, "
                  f"s={s_mm:4.1f}mm, v={v:4.1f}m/s")
        
        return False  # Continue optimization
    
    # Run optimization
    if verbose:
        print("\nStarting optimization...")
        print("-" * 70)
        print("  Iter     Objective  | Design Variables")
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
        polish=config.optimization.polish,  # Final local refinement
        workers=config.optimization.workers,  # Parallel evaluation
        updating=config.optimization.updating,
        atol=config.optimization.atol,
        disp=False
    )
    
    elapsed_time = time.time() - start_time
    
    x_opt = result.x
    x_opt[0] = int(round(x_opt[0]))  # Ensure N is integer
    J_opt = result.fun
    
    if verbose:
        print("-" * 70)
        print(f"\n✓ Optimization completed in {elapsed_time:.2f} seconds")
        print(f"  Function evaluations: {result.nfev}")
        print(f"  Iterations: {result.nit}")
        print(f"  Success: {result.success}")
        if not result.success:
            print(f"  Warning: {result.message}")
    
    # Package results
    result_dict = {
        'x_opt': x_opt,
        'J_opt': J_opt,
        'success': result.success,
        'message': result.message,
        'nfev': result.nfev,
        'nit': result.nit,
        'elapsed_time': elapsed_time,
        'convergence_history': best_values,
        'design_history': best_designs
    }
    
    return x_opt, J_opt, result_dict


def evaluate_design(x: np.ndarray, verbose: bool = True) -> Dict:
    """
    Evaluate a design and return all performance metrics with validation.
    
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
    
    # Add validation checks
    errors = []
    warnings = []
    
    # Physics validation
    Re = metrics['reynolds_number']
    aspect_ratio = (H * 1000) / (t * 1000)
    P_fan = metrics['fan_power']
    delta_P = metrics['pressure_drop']
    
    # Critical errors
    if T_base > config.operating.T_max:
        errors.append(f"Temperature {T_base:.1f}°C exceeds {config.operating.T_max}°C limit")
    
    if P_fan / (v**3) < 0.001:
        errors.append(f"Fan power {P_fan:.3f}W is unrealistically low for velocity {v:.1f}m/s")
    
    # Warnings
    if s * 1000 < 3.0:
        warnings.append(f"Spacing {s*1000:.1f}mm < 3mm may accumulate dust and restrict airflow")
    
    if aspect_ratio > 25:
        warnings.append(f"Aspect ratio {aspect_ratio:.1f} may cause fin vibration or buckling")
    
    if Re < 500:
        warnings.append(f"Low Reynolds number {Re:.0f} indicates poor convective heat transfer")
    
    if v > 6.0:
        noise_dB = 40 + 10 * np.log10(v / 2.0)
        warnings.append(f"High velocity {v:.1f}m/s may produce {noise_dB:.0f}dB noise")
    
    if delta_P > 500:
        warnings.append(f"High pressure drop {delta_P:.0f}Pa requires powerful fan")
    
    metrics['validation'] = {
        'errors': errors,
        'warnings': warnings,
        'aspect_ratio': aspect_ratio
    }
    
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
        print(f"  Reynolds number: {Re:.0f}")
        print(f"  Flow regime: {metrics['flow_regime']}")
        print(f"  Pressure drop: {delta_P:.2f} Pa")
        print(f"  Fan power: {P_fan:.3f} W")
        
        print(f"\nMaterial & Cost:")
        print(f"  Total mass: {metrics['total_mass']*1000:.2f} g")
        print(f"  Material cost: ${metrics['material_cost']:.2f}")
        
        print(f"\nFigures of Merit:")
        print(f"  Specific cooling: {metrics['specific_cooling']:.2f} W/kg")
        print(f"  Cost per watt: ${metrics['cost_per_watt']:.4f}/W")
        print(f"  Total system power: {metrics['total_system_power']:.2f} W")
        
        print(f"\nGeometry Check:")
        print(f"  Aspect ratio (H/t): {aspect_ratio:.1f}")
        print(f"  Total width: {(N*t + (N-1)*s)*1000:.1f} mm")
        
        # Validation report
        if errors or warnings:
            print(f"\n{'='*70}")
            print("VALIDATION REPORT")
            print(f"{'='*70}")
            
            if errors:
                print("\n❌ ERRORS (Design Invalid):")
                for err in errors:
                    print(f"   • {err}")
            
            if warnings:
                print("\n⚠️  WARNINGS:")
                for warn in warnings:
                    print(f"   • {warn}")
        else:
            print(f"\n✓ All validation checks passed")
        
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