"""
Configuration Module for heat sink optimization

This module contains all configuration parameters, material properties,
and design constraints for the heat sink optimization project.
"""

import numpy as np
from typing import Dict, Tuple 
from dataclasses import dataclass


@dataclass
class MaterialProperties:
    #Material properties for heat sink materials
    
    #Aluminium
    aluminium_k : float = 205.00  #Thermal conductivity
    aluminium_density : float =  2700.00   #Density
    aluminium_cost : float  = 2.50  #Cost per kg 
    
    #Copper 
    copper_k: float = 385.00              #Thermal conductivity
    copper_density: float = 8960.00          #Density
    copper_cost: float = 8.50             #Cost per kg
    
    
    #Air properties at 25°C
    air_k: float = 0.026   #Thermal conductivity 
    air_density: float =  1.184   #density
    air_kinematic_viscosity: float = 1.562e-5  #kinematic viscosity
    air_dynamic_viscosity: float = 1.849e-5    #dynamic viscosity
    air_prandtl: float = 0.707  #prandtl number
    air_cp: float = 1005.0  #Specific heat
    
@dataclass
class DesignBounds:
    """Design variiable bounds for optimization"""
    
      # Number of fins
    N_min: int = 5
    N_max: int = 50
    
    # Fin height [m]
    H_min: float = 0.01  # 10 mm
    H_max: float = 0.10  # 100 mm
    
    # Fin thickness [m]
    t_min: float = 0.001  # 1 mm
    t_max: float = 0.005  # 5 mm
    
    # Fin spacing [m]
    s_min: float = 0.002  # 2 mm
    s_max: float = 0.020  # 20 mm
    
    # Air velocity [m/s]
    v_min: float = 1.0
    v_max: float = 10.0
    
    def get_bounds_array(self) -> list:
        """Return bounds as list of tuples for scipy.optimize."""
        return [
            (self.N_min, self.N_max),
            (self.H_min, self.H_max),
            (self.t_min, self.t_max),
            (self.s_min, self.s_max),
            (self.v_min, self.v_max)
        ]
    
@dataclass
class OperatingConditions:
    """Operating conditions for the heat sink."""
    
    # Heat load [W]
    Q_cpu: float = 100.0  # Typical CPU heat dissipation
    
    # Ambient temperature [°C]
    T_ambient: float = 25.0
    
    # Maximum allowable base temperature [°C]
    T_max: float = 85.0  # Typical CPU thermal limit
    
    # Target base temperature [°C]
    T_target: float = 70.0  # Desired operating temperature
    
    # Heat sink base dimensions [m]
    base_width: float = 0.05  # 50 mm
    base_length: float = 0.05  # 50 mm
    base_thickness: float = 0.003  # 3 mm


@dataclass
class OptimizationParameters:
    """Parameters for differential evolution optimization."""
    
    # Differential Evolution parameters
    strategy: str = 'best1bin'  # DE strategy
    maxiter: int = 200  # Maximum iterations
    popsize: int = 15  # Population size multiplier
    tol: float = 0.01  # Convergence tolerance
    mutation: Tuple[float, float] = (0.5, 1.0)  # Mutation factor range
    recombination: float = 0.7  # Crossover probability
    seed: int = 42  # Random seed for reproducibility
    workers: int = -1  # Use all CPU cores (-1)
    polish: bool = True  # Local refinement of best solution
    atol: float = 0  # Absolute tolerance
    updating: str = 'deferred'  # Update strategy
    
    # Objective function weights
    weight_temperature: float = 1.0  # Weight for temperature objective
    weight_power: float = 0.5  # Weight for fan power objective
    weight_cost: float = 0.3  # Weight for material cost objective


@dataclass
class PhysicalConstraints:
    """Physical and manufacturing constraints."""
    
    # Minimum fin spacing constraint (manufacturability)
    spacing_multiplier: float = 2.0  # s >= 2*t
    
    # Maximum heat sink width [m]
    max_total_width: float = 0.15  # 150 mm
    
    # Fan efficiency
    fan_efficiency: float = 0.6  # Typical fan efficiency
    
    # Flow regime limits
    Re_laminar_limit: float = 2300.0  # Laminar to turbulent transition
    Re_turbulent_min: float = 3000.0  # Minimum Re for turbulent correlations


class Config:
    """Main configuration class combining all parameters."""
    
    def __init__(self):
        self.materials = MaterialProperties()
        self.bounds = DesignBounds()
        self.operating = OperatingConditions()
        self.optimization = OptimizationParameters()
        self.constraints = PhysicalConstraints()
    
    def get_design_variable_names(self) -> list:
        """Return list of design variable names."""
        return ['N', 'H', 't', 's', 'v']
    
    def get_design_variable_units(self) -> list:
        """Return list of design variable units."""
        return ['-', 'm', 'm', 'm', 'm/s']
    
    def validate_design(self, x: np.ndarray) -> Tuple[bool, str]:
        """
        Validate a design vector against constraints.
        
        Args:
            x: Design vector [N, H, t, s, v]
            
        Returns:
            (is_valid, message): Validation result and message
        """
        N, H, t, s, v = x
        
        # Check spacing constraint
        if s < self.constraints.spacing_multiplier * t:
            return False, f"Spacing {s:.4f} < 2*thickness {2*t:.4f}"
        
        # Check maximum width constraint
        total_width = N * (t + s)
        if total_width > self.constraints.max_total_width:
            return False, f"Total width {total_width:.4f} > max {self.constraints.max_total_width}"
        
        # Check bounds
        bounds = self.bounds.get_bounds_array()
        for i, (val, (lb, ub)) in enumerate(zip(x, bounds)):
            if val < lb or val > ub:
                var_names = self.get_design_variable_names()
                return False, f"{var_names[i]} = {val:.4f} outside bounds [{lb}, {ub}]"
        
        return True, "Valid design"
    
    def get_baseline_design(self) -> np.ndarray:
        """
        Return a conservative baseline design for comparison.
        
        Returns:
            x_baseline: Baseline design vector [N, H, t, s, v]
        """
        return np.array([
            20,      # N: 20 fins (conservative)
            0.040,   # H: 40 mm height
            0.002,   # t: 2 mm thickness
            0.005,   # s: 5 mm spacing
            3.0      # v: 3 m/s air velocity
        ])
    
    def print_configuration(self):
        """Print current configuration to console."""
        print("=" * 70)
        print("HEAT SINK OPTIMIZATION CONFIGURATION")
        print("=" * 70)
        print(f"\nMaterial Properties:")
        print(f"  Aluminum k: {self.materials.aluminum_k} W/m·K")
        print(f"  Aluminum density: {self.materials.aluminum_density} kg/m³")
        print(f"  Air k: {self.materials.air_k} W/m·K")
        
        print(f"\nOperating Conditions:")
        print(f"  Heat load: {self.operating.Q_cpu} W")
        print(f"  Ambient temp: {self.operating.T_ambient} °C")
        print(f"  Max base temp: {self.operating.T_max} °C")
        
        print(f"\nDesign Variable Bounds:")
        print(f"  Number of fins: {self.bounds.N_min} - {self.bounds.N_max}")
        print(f"  Fin height: {self.bounds.H_min*1000} - {self.bounds.H_max*1000} mm")
        print(f"  Fin thickness: {self.bounds.t_min*1000} - {self.bounds.t_max*1000} mm")
        print(f"  Fin spacing: {self.bounds.s_min*1000} - {self.bounds.s_max*1000} mm")
        print(f"  Air velocity: {self.bounds.v_min} - {self.bounds.v_max} m/s")
        
        print(f"\nOptimization Parameters:")
        print(f"  Algorithm: Differential Evolution")
        print(f"  Strategy: {self.optimization.strategy}")
        print(f"  Max iterations: {self.optimization.maxiter}")
        print(f"  Population size: {self.optimization.popsize} × dimensions")
        
        print("=" * 70)


# Create global configuration instance
config = Config()


if __name__ == "__main__":
    # Test configuration
    config.print_configuration()
    
    # Test baseline design
    x_baseline = config.get_baseline_design()
    is_valid, msg = config.validate_design(x_baseline)
    print(f"\nBaseline design validation: {is_valid}")
    print(f"Message: {msg}")
    print(f"Baseline design: N={x_baseline[0]:.0f}, H={x_baseline[1]*1000:.1f}mm, "
          f"t={x_baseline[2]*1000:.1f}mm, s={x_baseline[3]*1000:.1f}mm, v={x_baseline[4]:.1f}m/s")