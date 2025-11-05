"""
Performance metrics module for Heat Sink Optimization

This module calculates various performance metrics including :-
- Material mass and cost 
- Thermal resistance 
- Specific cooling power 
- Cost-performance ratios
"""

import numpy as np 
from typing import Dict
from config import config
from fluid_dynamics import (calculate_reynolds_number, calculate_hydraulic_diameter,
                            get_flow_regime, calculate_pressure_drop, calculate_fan_power)

def calculate_material_mass(N:int, H:float, t:float) -> float:
    """ 
    Calculate total mass of aluminum heat sink.
    
    Mass = ρ * Volume
    Volume = V_base + N * V_fin
    
    Args:
        N: Number of fins [-]
        H: Fin height [m]
        t: Fin thickness [m]
        
    Returns:
        mass: Total mass [kg]
    """ 
    rho = config.materials.aluminium_density
    W = config.operating.base_width
    L = config.operating.base_length
    t_base = config.operating.base_thickness
    
    # Base volume
    V_base = W * L * t_base
    
    # Single fin volume
    V_fin = W * H * t
    
    # Total volume
    V_total = V_base + N * V_fin
    
    # Total mass
    mass = rho * V_total
    
    return mass


def calculate_material_cost(mass:float) -> float:
    """
    Calculate material cost based on mass.
    
    Args:
        mass: Total mass [kg]
        
    Returns:
        cost: Material cost [$]
    """
    cost_per_kg = config.materials.aluminium_cost
    cost = mass * cost_per_kg
    return cost

def calculate_thermal_resistance_full(T_base:float, T_inf:float, Q: float) -> float:
    """
    Calculate thermal resistance.
    
    Args:
        T_base: Base temperature [°C]
        T_inf: Ambient temperature [°C]
        Q: Heat dissipation [W]
        
    Returns:
        R_th: Thermal resistance [K/W]
    """
    if Q < 1e-6:
        return np.inf
    
    R_th = (T_base - T_inf) / Q
    return R_th


def calculate_specific_cooling_power(Q: float, mass: float) -> float:
    """
    Calculate specific cooling power (heat dissipated per unit mass).
    
    Args:
        Q: Heat dissipation [W]
        mass: Total mass [kg]
        
    Returns:
        specific_power: Specific cooling power [W/kg]
    """
    if mass < 1e-9:
        return 0.0
    
    specific_power = Q / mass
    return specific_power


def calculate_cost_per_watt(cost: float, Q: float) -> float:
    """
    Calculate cost per watt of cooling capacity.
    
    Args:
        cost: Material cost [$]
        Q: Heat dissipation [W]
        
    Returns:
        cost_per_watt: Cost per watt [$/W]
    """
    if Q < 1e-6:
        return np.inf
    
    cost_per_watt = cost / Q
    return cost_per_watt


def calculate_total_system_power(P_fan: float, Q_cpu: float) -> float:
    """
    Calculate total system power (CPU + fan).
    
    Args:
        P_fan: Fan power [W]
        Q_cpu: CPU heat dissipation [W]
        
    Returns:
        P_total: Total power [W]
    """
    P_total = Q_cpu + P_fan
    return P_total


def calculate_cooling_efficiency(Q_cpu: float, P_fan: float) -> float:
    """
    Calculate cooling efficiency (ratio of heat dissipated to fan power).
    
    Higher is better - want to dissipate more heat with less fan power.
    
    Args:
        Q_cpu: CPU heat dissipation [W]
        P_fan: Fan power [W]
        
    Returns:
        efficiency: Cooling efficiency [-]
    """
    if P_fan < 1e-6:
        return np.inf
    
    efficiency = Q_cpu / P_fan
    return efficiency


def calculate_pec(T_base: float, T_inf: float, Q_cpu: float, P_fan: float) -> float:
    """
    Calculate Performance Evaluation Criterion (PEC).
    
    PEC combines thermal performance and power consumption.
    PEC = (Q / ΔT) / P_fan
    
    Higher is better.
    
    Args:
        T_base: Base temperature [°C]
        T_inf: Ambient temperature [°C]
        Q_cpu: CPU heat dissipation [W]
        P_fan: Fan power [W]
        
    Returns:
        pec: Performance evaluation criterion [W/(K·W)]
    """
    delta_T = T_base - T_inf
    
    if delta_T < 1e-3 or P_fan < 1e-6:
        return 0.0
    
    pec = (Q_cpu / delta_T) / P_fan
    return pec


def calculate_all_metrics(x: np.ndarray, T_base: float, info: Dict) -> Dict:
    """
    Calculate all performance metrics for a given design.
    
    Args:
        x: Design vector [N, H, t, s, v]
        T_base: Base temperature [°C]
        info: Information from temperature solver
        
    Returns:
        metrics: Dictionary containing all performance metrics
    """
    N = int(round(x[0]))
    H = x[1]
    t = x[2]
    s = x[3]
    v = x[4]
    
    Q_cpu = config.operating.Q_cpu
    T_inf = config.operating.T_ambient
    W = config.operating.base_width
    
    # Calculate mass and cost
    mass = calculate_material_mass(N, H, t)
    cost = calculate_material_cost(mass)
    
    # Thermal metrics
    R_th = calculate_thermal_resistance_full(T_base, T_inf, Q_cpu)
    
    # Fluid dynamics metrics
    T_film = (T_base + T_inf) / 2.0
    D_h = calculate_hydraulic_diameter(s, W)
    Re = calculate_reynolds_number(v, D_h, T_film)
    flow_regime = get_flow_regime(Re)
    delta_P = calculate_pressure_drop(N, s, H, v, T_film)
    P_fan = calculate_fan_power(delta_P, v, N, s)
    
    # Derived metrics
    specific_cooling = calculate_specific_cooling_power(Q_cpu, mass)
    cost_per_watt = calculate_cost_per_watt(cost, Q_cpu)
    P_total = calculate_total_system_power(P_fan, Q_cpu)
    cooling_eff = calculate_cooling_efficiency(Q_cpu, P_fan)
    pec = calculate_pec(T_base, T_inf, Q_cpu, P_fan)
    
    # Compile all metrics
    metrics = {
        # Design parameters
        'N': N,
        'H': H,
        't': t,
        's': s,
        'v': v,
        
        # Thermal performance
        'T_base': T_base,
        'T_inf': T_inf,
        'delta_T': T_base - T_inf,
        'Q_cpu': Q_cpu,
        'thermal_resistance': R_th,
        'convection_coeff': info.get('h', 0),
        
        # Fluid dynamics
        'reynolds_number': Re,
        'flow_regime': flow_regime,
        'pressure_drop': delta_P,
        'fan_power': P_fan,
        
        # Material and cost
        'total_mass': mass,
        'material_cost': cost,
        
        # Derived metrics
        'specific_cooling': specific_cooling,
        'cost_per_watt': cost_per_watt,
        'total_system_power': P_total,
        'cooling_efficiency': cooling_eff,
        'pec': pec,
        
        # Solver info
        'converged': info.get('converged', False),
        'iterations': info.get('iterations', 0)
    }
    
    return metrics


def compare_designs(designs: Dict[str, np.ndarray], verbose: bool = True) -> Dict:
    """
    Compare multiple heat sink designs.
    
    Args:
        designs: Dictionary of {name: design_vector}
        verbose: Print comparison table
        
    Returns:
        comparison: Dictionary of design metrics
    """
    from heat_transfer import solve_base_temperature
    
    Q_load = config.operating.Q_cpu
    T_inf = config.operating.T_ambient
    
    comparison = {}
    
    for name, x in designs.items():
        N = int(round(x[0]))
        H = x[1]
        t = x[2]
        s = x[3]
        v = x[4]
        
        # Solve for temperature
        T_base, info = solve_base_temperature(N, H, t, s, v, Q_load, T_inf)
        
        # Calculate metrics
        metrics = calculate_all_metrics(x, T_base, info)
        comparison[name] = metrics
    
    if verbose:
        print("\n" + "="*90)
        print("DESIGN COMPARISON")
        print("="*90)
        
        # Print header
        print(f"\n{'Metric':<25} ", end="")
        for name in designs.keys():
            print(f"{name:>15} ", end="")
        print()
        print("-" * 90)
        
        # Key metrics to compare
        key_metrics = [
            ('T_base', '°C', '.2f'),
            ('thermal_resistance', 'K/W', '.4f'),
            ('fan_power', 'W', '.3f'),
            ('pressure_drop', 'Pa', '.2f'),
            ('total_mass', 'kg', '.4f'),
            ('material_cost', '$', '.2f'),
            ('specific_cooling', 'W/kg', '.1f'),
            ('cost_per_watt', '$/W', '.4f'),
            ('cooling_efficiency', '-', '.1f'),
            ('pec', 'W/(K·W)', '.2f')
        ]
        
        for metric_name, unit, fmt in key_metrics:
            print(f"{metric_name:<25} ", end="")
            for name in designs.keys():
                value = comparison[name][metric_name]
                print(f"{value:{fmt}:>15} ", end="")
            print()
        
        print("="*90)
    
    return comparison


def rank_designs(comparison: Dict, criteria: str = 'pec') -> list:
    """
    Rank designs based on a specific criterion.
    
    Args:
        comparison: Dictionary from compare_designs
        criteria: Metric to rank by ('pec', 'thermal_resistance', etc.)
        
    Returns:
        ranking: List of (name, value) tuples sorted by criterion
    """
    ranking = []
    
    for name, metrics in comparison.items():
        value = metrics.get(criteria, 0)
        ranking.append((name, value))
    
    # Sort based on criterion (lower is better for most metrics except pec, cooling_efficiency)
    if criteria in ['pec', 'cooling_efficiency', 'specific_cooling']:
        ranking.sort(key=lambda x: x[1], reverse=True)  # Higher is better
    else:
        ranking.sort(key=lambda x: x[1])  # Lower is better
    
    return ranking


if __name__ == "__main__":
    # Test performance calculations
    print("Testing performance metrics module...")
    
    # Test design
    x_test = np.array([20, 0.05, 0.002, 0.004, 4.0])
    
    from heat_transfer import solve_base_temperature
    
    Q_load = config.operating.Q_cpu
    T_inf = config.operating.T_ambient
    
    T_base, info = solve_base_temperature(
        int(x_test[0]), x_test[1], x_test[2], x_test[3], x_test[4],
        Q_load, T_inf
    )
    
    metrics = calculate_all_metrics(x_test, T_base, info)
    
    print("\nTest Design Metrics:")
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.number)):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")
