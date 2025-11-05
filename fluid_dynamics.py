"""
    Fluid dynamics module
    
    This module implements following fluid flow calculations:-
    - Reynolds number calculation
    - Nusselt number calculation
    - Convection coefficient determination
    - Pressure drop calculations
    - Fan power requirements
""" 

import numpy as np
from typing import Dict
from config import config

def calculate_hydraulic_diameter(s: float, W: float) -> float:
    """
    Calculate hydraulic diameter for flow between parallel plates(fins).
    
    For parallel plates with spacing s:
    D_h = 4 * A_flow/P_wetted = 4 * (s*W)/(2*W) = 2*s 
    
    Args:
    s: Fin spacing[m]
    W: Channel width [m]
    
    Returns:
    D_h : Hydraulic diameter [m]
    """
    
    #For parallel plate channel
    D_h = 2*s
    return D_h

def calculate_reynolds_number(v:float, D_h:float, T_film: float) -> float:
    """_Calculate Reynolds number for flow between fins.
    
    Re = ρ * v * D_h / μ = v * D_h / ν
    
    Args:
        v: Air velocity [m/s]
        D_h: Hydraulic diameter [m]
        T_film: Film temperature for property evaluation [°C]
        
    Returns:
        Re: Reynolds number [-]
    """
    
    #Air properties(assuming small temperature dependence)
    nu = config.materials.air_kinematic_viscosity
    
    #Temperature correction(approximate)
    # ν ∝ T^0.7 for air
    T_ref = 25.0  # Reference temperature [°C]
    nu_corrected = nu * ((T_film + 273.15) / (T_ref + 273.15))**0.7
    
    Re = v * D_h / nu_corrected
    return Re


def calculate_nusselt_number(Re: float, Pr: float, flow_length: float, D_h: float) -> float:
    """
     Calculate Nusselt number using appropriate correlations.
    
    For internal flow between parallel plates:
    - Laminar (Re < 2300): Nu = 7.54 (fully developed, constant heat flux)
    - Turbulent (Re > 3000): Nu = 0.023 * Re^0.8 * Pr^0.4 (Dittus-Boelter)
    - Transition (2300 < Re < 3000): Linear interpolation
    
    Args:
        Re: Reynolds number [-]
        Pr: Prandtl number [-]
        flow_length: Length of flow channel [m]
        D_h: Hydraulic diameter [m]
        
    Returns:
        Nu: Nusselt number [-]
    """
    
    #Check the flow regime
    Re_lamb_limit = config.constraints.Re_laminar_limit
    Re_turb_min = config.constraints.Re_turbulent_min
    
    
    if Re < Re_lamb_limit:
        #Laminar flow
        #For parallel plate with constant heat flux
        Nu  = 7.54
        
        #if Length correction needed
        L_entrance  = 0.05 * Re * D_h
        
        if flow_length < L_entrance:
            #Developing flow use average Nusselt number 
            Nu = Nu * 1.12   #Approximate correction
            
    elif Re > Re_turb_min:
        #Turbulent flow
        Nu = 0.023 * (Re**0.8) * (Pr**0.4)  #Dittus - Boelter equation
        
        #Entrance effects (Minor for turbulent flow)
        L_entrance = 10 * D_h
        if flow_length < L_entrance:
            Nu = Nu * 1.1
            
    else:
        # Transition regime - linear interpolation
        Nu_lam = 7.54
        Nu_turb = 0.023 * (Re_turb_min**0.8) * (Pr**0.4)
        
        #Linear interpolation
        weight = (Re - Re_lamb_limit)/(Re_turb_min - Re_lamb_limit)
        Nu = Nu_lam + weight*(Nu_turb - Nu_lam)
        
    return Nu

def calculate_convection_coefficient(v: float, s: float, H: float,
                                     T_surface: float, T_inf: float) -> float:
    """
    Calculate convection heat transfer coefficient.
    
    h = Nu * k_air / D_h
    
    Args:
        v: Air velocity [m/s]
        s: Fin spacing [m]
        H: Fin height (flow length) [m]
        T_surface: Surface temperature [°C]
        T_inf: Ambient air temperature [°C]
        
    Returns:
        h: Convection coefficient [W/m²·K]
    """
    # Film temperature for property evaluation
    T_film = (T_surface + T_inf) / 2.0
    
    # Hydraulic diameter
    W = config.operating.base_width
    D_h = calculate_hydraulic_diameter(s, W)
    
    # Reynolds number
    Re = calculate_reynolds_number(v, D_h, T_film)
    
    # Prandtl number (approximately constant for air)
    Pr = config.materials.air_prandtl
    
    # Nusselt number
    Nu = calculate_nusselt_number(Re, Pr, H, D_h)
    
    # Air thermal conductivity (with temperature correction)
    k_air = config.materials.air_k
    T_ref = 25.0
    k_air_corrected = k_air * ((T_film + 273.15) / (T_ref + 273.15))**0.8
    
    # Convection coefficient
    h = Nu * k_air_corrected / D_h
    
    return h


def calculate_friction_factor(Re: float) -> float:
    """
    Calculate Darcy friction factor for internal flow.
    
    For parallel plates:
    - Laminar (Re < 2300): f = 96 / Re (for parallel plates, not pipes!)
    - Turbulent (Re > 3000): f = 0.079 / Re^0.25 (Blasius correlation)
    - Transition: interpolation
    
    Args:
        Re: Reynolds number [-]
        
    Returns:
        f: Darcy friction factor [-]
    """
    Re_lam_limit = config.constraints.Re_laminar_limit
    Re_turb_min = config.constraints.Re_turbulent_min
    
    if Re < Re_lam_limit:
        # Laminar flow between parallel plates
        f = 96.0 / Re
    
    elif Re > Re_turb_min:
        # Turbulent flow - Blasius correlation
        f = 0.079 / (Re**0.25)
    
    else:
        # Transition region
        f_lam = 96.0 / Re
        f_turb = 0.079 / (Re**0.25)
        
        weight = (Re - Re_lam_limit) / (Re_turb_min - Re_lam_limit)
        f = f_lam + weight * (f_turb - f_lam)
    
    return f


def calculate_pressure_drop(N: int, s: float, H: float, v: float,
                           T_film: float) -> float:
    """
    Calculate pressure drop through fin array.
    
    ΔP = f * (L / D_h) * (ρ * v² / 2)
    
    Args:
        N: Number of fins [-]
        s: Fin spacing [m]
        H: Fin height (flow length) [m]
        v: Air velocity [m/s]
        T_film: Film temperature [°C]
        
    Returns:
        delta_P: Pressure drop [Pa]
    """
    W = config.operating.base_width
    
    # Hydraulic diameter
    D_h = calculate_hydraulic_diameter(s, W)
    
    # Reynolds number
    Re = calculate_reynolds_number(v, D_h, T_film)
    
    # Friction factor
    f = calculate_friction_factor(Re)
    
    # Air density (with temperature correction)
    rho = config.materials.air_density
    T_ref = 25.0
    rho_corrected = rho * (T_ref + 273.15) / (T_film + 273.15)
    
    # Pressure drop in one channel
    delta_P = f * (H / D_h) * (rho_corrected * v**2 / 2.0)
    
    return delta_P


def calculate_fan_power(delta_P: float, v: float, N: int, s: float) -> float:
    """
    Calculate fan power consumption.
    
    P_fan = (ΔP * V̇) / η_fan
    where V̇ = v * A_flow
    
    Args:
        delta_P: Pressure drop [Pa]
        v: Air velocity [m/s]
        N: Number of fins [-]
        s: Fin spacing [m]
        
    Returns:
        P_fan: Fan power [W]
    """
    W = config.operating.base_width
    
    # Number of flow channels (between fins)
    n_channels = N - 1 if N > 1 else 1
    
    # Flow area
    A_flow = n_channels * s * W
    
    # Volumetric flow rate
    V_dot = v * A_flow
    
    # Fan efficiency
    eta_fan = config.constraints.fan_efficiency
    
    # Fan power
    P_fan = (delta_P * V_dot) / eta_fan
    
    return P_fan


def get_flow_regime(Re: float) -> str:
    """
    Determine flow regime based on Reynolds number.
    
    Args:
        Re: Reynolds number [-]
        
    Returns:
        regime: Flow regime ('laminar', 'transition', 'turbulent')
    """
    Re_lam_limit = config.constraints.Re_laminar_limit
    Re_turb_min = config.constraints.Re_turbulent_min
    
    if Re < Re_lam_limit:
        return 'laminar'
    elif Re > Re_turb_min:
        return 'turbulent'
    else:
        return 'transition'


def validate_fluid_dynamics_calculations():
    """
    Validate fluid dynamics calculations with known correlations.
    """
    print("\n" + "="*70)
    print("VALIDATING FLUID DYNAMICS CALCULATIONS")
    print("="*70)
    
    # Test case 1: Reynolds number and flow regime
    v = 5.0  # m/s
    s = 0.004  # 4 mm spacing
    W = 0.05  # 50 mm width
    T_film = 40.0  # °C
    
    D_h = calculate_hydraulic_diameter(s, W)
    Re = calculate_reynolds_number(v, D_h, T_film)
    regime = get_flow_regime(Re)
    
    print(f"\nTest 1: Flow Regime Analysis")
    print(f"  Velocity: {v:.1f} m/s")
    print(f"  Fin spacing: {s*1000:.1f} mm")
    print(f"  Hydraulic diameter: {D_h*1000:.2f} mm")
    print(f"  Reynolds number: {Re:.0f}")
    print(f"  Flow regime: {regime}")
    
    # Test case 2: Nusselt number
    Pr = config.materials.air_prandtl
    H = 0.05  # 50 mm flow length
    Nu = calculate_nusselt_number(Re, Pr, H, D_h)
    
    print(f"\nTest 2: Nusselt Number")
    print(f"  Reynolds: {Re:.0f}")
    print(f"  Prandtl: {Pr:.3f}")
    print(f"  Nusselt: {Nu:.2f}")
    
    # Expected values for verification
    if regime == 'laminar':
        print(f"  Expected Nu ≈ 7.54 (fully developed)")
    else:
        Nu_expected = 0.023 * (Re**0.8) * (Pr**0.4)
        print(f"  Expected Nu ≈ {Nu_expected:.2f} (Dittus-Boelter)")
    
    # Test case 3: Convection coefficient
    h = calculate_convection_coefficient(v, s, H, T_film + 10, T_film - 10)
    print(f"\nTest 3: Convection Coefficient")
    print(f"  h = {h:.2f} W/m²·K")
    print(f"  Typical range for forced air: 20-100 W/m²·K")
    
    # Test case 4: Pressure drop and fan power
    N = 20
    delta_P = calculate_pressure_drop(N, s, H, v, T_film)
    P_fan = calculate_fan_power(delta_P, v, N, s)
    
    print(f"\nTest 4: Pressure Drop and Fan Power")
    print(f"  Number of fins: {N}")
    print(f"  Pressure drop: {delta_P:.2f} Pa")
    print(f"  Fan power: {P_fan:.3f} W")
    
    # Friction factor
    f = calculate_friction_factor(Re)
    print(f"  Friction factor: {f:.5f}")
    
    print("="*70)


if __name__ == "__main__":
    # Run validation
    validate_fluid_dynamics_calculations()
