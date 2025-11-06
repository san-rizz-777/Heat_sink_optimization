"""
Utility Functions Module for Heat Sink Optimization
Course: Transport Phenomena
Author: Sanskar Gunde (MM24B005)
Date: November 2025

This module provides utility functions for:
- File I/O operations
- Result formatting and saving
- Directory management
- Helper functions
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any


def print_banner():
    """Print project banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║     HEAT SINK OPTIMIZATION FOR DATA CENTER COOLING               ║
    ║                                                                  ║
    ║     Transport Phenomena Programming Project                      ║
    ║     Differential Evolution Optimization                          ║
    ║                                                                  ║
    ║     Author: Sanskar Gunde (MM24B005)                            ║
    ║     Date: November 2025                                          ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def create_results_directory(timestamp: str = None) -> str:
    """
    Create a timestamped results directory.
    
    Args:
        timestamp: Optional timestamp string. If None, current time is used.
        
    Returns:
        dir_path: Path to created directory
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create main results directory
    base_dir = "./results"
    os.makedirs(base_dir, exist_ok=True)
    
    # Create timestamped subdirectory
    dir_path = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(dir_path, exist_ok=True)
    
    # Create subdirectories for plots and data
    os.makedirs(os.path.join(dir_path, "plots"), exist_ok=True)
    os.makedirs(os.path.join(dir_path, "data"), exist_ok=True)
    
    print(f"\nCreated results directory: {dir_path}")
    
    return dir_path


def save_results_to_file(results_data: Dict, output_dir: str) -> str:
    """
    Save optimization results to JSON file.
    
    Args:
        results_data: Dictionary containing all results
        output_dir: Directory to save file
        
    Returns:
        filepath: Path to saved file
    """
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_data = convert_to_serializable(results_data)
    
    # Save to JSON file
    filepath = os.path.join(output_dir, "optimization_results.json")
    with open(filepath, 'w') as f:
        json.dump(serializable_data, f, indent=4)
    
    # Also save a human-readable summary
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("HEAT SINK OPTIMIZATION RESULTS SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Run Timestamp: {results_data['timestamp']}\n\n")
        
        f.write("BASELINE DESIGN:\n")
        x_baseline = results_data['baseline_design']
        f.write(f"  N = {int(round(x_baseline[0]))}, ")
        f.write(f"H = {x_baseline[1]*1000:.2f} mm, ")
        f.write(f"t = {x_baseline[2]*1000:.2f} mm, ")
        f.write(f"s = {x_baseline[3]*1000:.2f} mm, ")
        f.write(f"v = {x_baseline[4]:.2f} m/s\n")
        
        metrics_b = results_data['baseline_metrics']
        f.write(f"  T_base = {metrics_b['T_base']:.2f}°C\n")
        f.write(f"  P_fan = {metrics_b['fan_power']:.3f} W\n")
        f.write(f"  Cost = ${metrics_b['material_cost']:.2f}\n\n")
        
        f.write("OPTIMIZED DESIGN:\n")
        x_opt = results_data['optimized_design']
        f.write(f"  N = {int(round(x_opt[0]))}, ")
        f.write(f"H = {x_opt[1]*1000:.2f} mm, ")
        f.write(f"t = {x_opt[2]*1000:.2f} mm, ")
        f.write(f"s = {x_opt[3]*1000:.2f} mm, ")
        f.write(f"v = {x_opt[4]:.2f} m/s\n")
        
        metrics_o = results_data['optimized_metrics']
        f.write(f"  T_base = {metrics_o['T_base']:.2f}°C\n")
        f.write(f"  P_fan = {metrics_o['fan_power']:.3f} W\n")
        f.write(f"  Cost = ${metrics_o['material_cost']:.2f}\n\n")
        
        f.write("IMPROVEMENTS:\n")
        imp = results_data['improvements']
        f.write(f"  Temperature: {imp['temperature']:.2f}°C\n")
        f.write(f"  Power: {imp['power']:.3f} W\n")
        f.write(f"  Cost: ${imp['cost']:.2f}\n")
        f.write(f"  Annual energy savings: {imp['annual_energy_savings_kwh']:.1f} kWh\n")
        f.write(f"  Annual cost savings: ${imp['annual_cost_savings']:.2f}\n\n")
        
        f.write("="*70 + "\n")
    
    print(f"Results saved to: {filepath}")
    print(f"Summary saved to: {summary_path}")
    
    return filepath


def format_design_vector(x: np.ndarray, include_units: bool = True) -> str:
    """
    Format design vector as readable string.
    
    Args:
        x: Design vector [N, H, t, s, v]
        include_units: Include units in output
        
    Returns:
        formatted: Formatted string
    """
    if include_units:
        return (f"N={int(round(x[0]))}, "
                f"H={x[1]*1000:.2f}mm, "
                f"t={x[2]*1000:.2f}mm, "
                f"s={x[3]*1000:.2f}mm, "
                f"v={x[4]:.2f}m/s")
    else:
        return (f"N={int(round(x[0]))}, "
                f"H={x[1]:.5f}, "
                f"t={x[2]:.5f}, "
                f"s={x[3]:.5f}, "
                f"v={x[4]:.3f}")


def validate_inputs(Q_cpu: float, T_ambient: float, T_max: float) -> bool:
    """
    Validate input parameters for physical feasibility.
    
    Args:
        Q_cpu: CPU heat load [W]
        T_ambient: Ambient temperature [°C]
        T_max: Maximum allowable temperature [°C]
        
    Returns:
        valid: True if inputs are valid
    """
    if Q_cpu <= 0:
        print("ERROR: Heat load must be positive")
        return False
    
    if Q_cpu > 500:
        print("WARNING: Heat load > 500W is very high for typical CPUs")
    
    if T_ambient < 0 or T_ambient > 50:
        print("WARNING: Ambient temperature outside typical range (0-50°C)")
    
    if T_max <= T_ambient:
        print("ERROR: Maximum temperature must be higher than ambient")
        return False
    
    if T_max > 100:
        print("WARNING: Maximum temperature > 100°C may damage components")
    
    return True


def calculate_fin_count_estimate(Q_cpu: float, T_max: float, T_ambient: float) -> int:
    """
    Rough estimate of required fin count based on heat load.
    
    This is a heuristic based on typical correlations.
    
    Args:
        Q_cpu: Heat load [W]
        T_max: Maximum temperature [°C]
        T_ambient: Ambient temperature [°C]
        
    Returns:
        N_estimate: Estimated fin count
    """
    # Typical convection coefficient for forced air: 50 W/m²·K
    h_typical = 50.0
    
    # Available temperature difference
    delta_T = T_max - T_ambient
    
    # Required heat transfer conductance [W/K]
    UA_required = Q_cpu / delta_T
    
    # Typical fin dimensions
    H_typical = 0.05  # 50 mm
    W_typical = 0.05  # 50 mm
    A_fin = 2 * H_typical * W_typical  # Two sides
    
    # Assuming 80% fin efficiency
    eta_typical = 0.8
    
    # Estimate fin count
    N_estimate = int(np.ceil(UA_required / (h_typical * A_fin * eta_typical)))
    
    # Practical bounds
    N_estimate = max(5, min(N_estimate, 50))
    
    return N_estimate


def print_progress_bar(iteration: int, total: int, prefix: str = '', 
                      suffix: str = '', length: int = 50):
    """
    Print a progress bar to console.
    
    Args:
        iteration: Current iteration
        total: Total iterations
        prefix: Prefix string
        suffix: Suffix string
        length: Character length of bar
    """
    percent = 100 * (iteration / float(total))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    
    print(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}', end='', flush=True)
    
    if iteration == total:
        print()


def load_results_from_file(filepath: str) -> Dict:
    """
    Load previously saved results from JSON file.
    
    Args:
        filepath: Path to results file
        
    Returns:
        results: Dictionary of results
    """
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    # Convert lists back to numpy arrays where needed
    if 'baseline_design' in results:
        results['baseline_design'] = np.array(results['baseline_design'])
    if 'optimized_design' in results:
        results['optimized_design'] = np.array(results['optimized_design'])
    
    return results


def export_to_csv(data: Dict, filepath: str):
    """
    Export results to CSV format for easy analysis in Excel/other tools.
    
    Args:
        data: Dictionary of results
        filepath: Path to save CSV file
    """
    import csv
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['Parameter', 'Value', 'Unit'])
        
        # Write data
        for key, value in data.items():
            if isinstance(value, (int, float, np.number)):
                writer.writerow([key, value, ''])
            elif isinstance(value, str):
                writer.writerow([key, value, ''])
    
    print(f"Data exported to CSV: {filepath}")


def compare_with_previous_runs(current_design: np.ndarray, 
                               results_dir: str = "./results") -> Dict:
    """
    Compare current optimization with previous runs.
    
    Args:
        current_design: Current optimized design
        results_dir: Directory containing previous results
        
    Returns:
        comparison: Dictionary comparing with historical runs
    """
    # This would scan previous result files and compare
    # Implementation depends on file structure
    pass


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    print_banner()
    
    # Test directory creation
    test_dir = create_results_directory("test")
    print(f"Created test directory: {test_dir}")
    
    # Test design formatting
    test_design = np.array([20, 0.05, 0.002, 0.004, 3.5])
    formatted = format_design_vector(test_design)
    print(f"Formatted design: {formatted}")
    
    # Test input validation
    valid = validate_inputs(100.0, 25.0, 85.0)
    print(f"Input validation: {valid}")
    
    # Test fin count estimate
    N_est = calculate_fin_count_estimate(100.0, 85.0, 25.0)
    print(f"Estimated fin count: {N_est}")
    
    print("\nUtility tests completed!")