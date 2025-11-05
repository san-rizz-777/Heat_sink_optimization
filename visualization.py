"""
Visualization Module for Heat Sink Optimization
Course: Transport Phenomena
Author: Sanskar Gunde (MM24B005)
Date: November 2025

This module creates all visualizations including:
- Convergence plots
- Parametric studies
- Performance comparisons
- Temperature distributions
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from matplotlib.patches import Rectangle

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_convergence(convergence_history: List[float], save_path: str = 'convergence.png'):
    """
    Plot optimization convergence history.
    
    Args:
        convergence_history: List of best objective values per iteration
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = list(range(1, len(convergence_history) + 1))
    ax.plot(iterations, convergence_history, 'b-', linewidth=2, label='Best Objective')
    
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Objective Function Value', fontsize=12, fontweight='bold')
    ax.set_title('Optimization Convergence History', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add improvement annotation
    improvement = (1 - convergence_history[-1]/convergence_history[0]) * 100
    ax.text(0.98, 0.95, f'Improvement: {improvement:.1f}%',
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Convergence plot saved to {save_path}")


def plot_design_comparison(comparison: Dict, save_path: str = 'design_comparison.png'):
    """
    Plot bar charts comparing different designs.
    
    Args:
        comparison: Dictionary from compare_designs or compare_with_literature
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    methods = list(comparison.keys())
    
    # Metric 1: Base Temperature
    temps = [comparison[m].get('T_base', 0) for m in methods]
    axes[0, 0].barh(methods, temps, color='orangered', alpha=0.7)
    axes[0, 0].set_xlabel('Base Temperature [°C]', fontweight='bold')
    axes[0, 0].set_title('Thermal Performance', fontweight='bold')
    axes[0, 0].axvline(x=85, color='red', linestyle='--', label='T_max limit')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Metric 2: Fan Power
    powers = [comparison[m].get('P_fan', comparison[m].get('fan_power', 0)) for m in methods]
    axes[0, 1].barh(methods, powers, color='steelblue', alpha=0.7)
    axes[0, 1].set_xlabel('Fan Power [W]', fontweight='bold')
    axes[0, 1].set_title('Power Consumption', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Metric 3: Material Cost
    costs = [comparison[m].get('material_cost', comparison[m].get('cost', 0)) for m in methods]
    axes[1, 0].barh(methods, costs, color='green', alpha=0.7)
    axes[1, 0].set_xlabel('Cost [$]', fontweight='bold')
    axes[1, 0].set_title('Material Cost', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Metric 4: PEC (Performance Evaluation Criterion)
    pecs = [comparison[m].get('pec', 0) for m in methods]
    axes[1, 1].barh(methods, pecs, color='purple', alpha=0.7)
    axes[1, 1].set_xlabel('PEC [W/(K·W)]', fontweight='bold')
    axes[1, 1].set_title('Performance Evaluation Criterion\n(Higher is Better)', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Design comparison plot saved to {save_path}")


def plot_parametric_study(param_name: str, param_values: np.ndarray,
                         results: Dict, save_path: str = 'parametric_study.png'):
    """
    Plot results of parametric study.
    
    Args:
        param_name: Name of parameter varied
        param_values: Array of parameter values
        results: Dictionary with arrays of results
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Temperature vs parameter
    axes[0, 0].plot(param_values, results['T_base'], 'o-', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel(param_name, fontweight='bold')
    axes[0, 0].set_ylabel('Base Temperature [°C]', fontweight='bold')
    axes[0, 0].set_title(f'Temperature vs {param_name}', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Fan power vs parameter
    axes[0, 1].plot(param_values, results['P_fan'], 'o-', linewidth=2, 
                   markersize=6, color='steelblue')
    axes[0, 1].set_xlabel(param_name, fontweight='bold')
    axes[0, 1].set_ylabel('Fan Power [W]', fontweight='bold')
    axes[0, 1].set_title(f'Fan Power vs {param_name}', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Thermal resistance vs parameter
    axes[1, 0].plot(param_values, results['R_th'], 'o-', linewidth=2,
                   markersize=6, color='red')
    axes[1, 0].set_xlabel(param_name, fontweight='bold')
    axes[1, 0].set_ylabel('Thermal Resistance [K/W]', fontweight='bold')
    axes[1, 0].set_title(f'Thermal Resistance vs {param_name}', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # PEC vs parameter
    axes[1, 1].plot(param_values, results['PEC'], 'o-', linewidth=2,
                   markersize=6, color='purple')
    axes[1, 1].set_xlabel(param_name, fontweight='bold')
    axes[1, 1].set_ylabel('PEC [W/(K·W)]', fontweight='bold')
    axes[1, 1].set_title(f'PEC vs {param_name}', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Parametric study plot saved to {save_path}")


def plot_pareto_frontier(designs: List[Tuple], save_path: str = 'pareto_frontier.png'):
    """
    Plot Pareto frontier for multi-objective optimization.
    
    Args:
        designs: List of (T_base, P_fan, cost) tuples
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(14, 10))
    
    # 2D projections
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    
    temps = [d[0] for d in designs]
    powers = [d[1] for d in designs]
    costs = [d[2] for d in designs]
    
    # Temperature vs Fan Power
    ax1.scatter(temps, powers, c='blue', s=50, alpha=0.6)
    ax1.set_xlabel('Base Temperature [°C]', fontweight='bold')
    ax1.set_ylabel('Fan Power [W]', fontweight='bold')
    ax1.set_title('Temperature vs Power Trade-off', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Temperature vs Cost
    ax2.scatter(temps, costs, c='green', s=50, alpha=0.6)
    ax2.set_xlabel('Base Temperature [°C]', fontweight='bold')
    ax2.set_ylabel('Cost [$]', fontweight='bold')
    ax2.set_title('Temperature vs Cost Trade-off', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Fan Power vs Cost
    ax3.scatter(powers, costs, c='red', s=50, alpha=0.6)
    ax3.set_xlabel('Fan Power [W]', fontweight='bold')
    ax3.set_ylabel('Cost [$]', fontweight='bold')
    ax3.set_title('Power vs Cost Trade-off', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 3D Pareto frontier
    ax4.scatter(temps, powers, costs, c='purple', s=50, alpha=0.6)
    ax4.set_xlabel('Temperature [°C]', fontweight='bold')
    ax4.set_ylabel('Fan Power [W]', fontweight='bold')
    ax4.set_zlabel('Cost [$]', fontweight='bold')
    ax4.set_title('3D Pareto Frontier', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Pareto frontier plot saved to {save_path}")


def plot_heat_sink_geometry(design: np.ndarray, save_path: str = 'geometry.png'):
    """
    Visualize heat sink geometry (2D schematic).
    
    Args:
        design: Design vector [N, H, t, s, v]
        save_path: Path to save figure
    """
    N = int(round(design[0]))
    H = design[1] * 1000  # Convert to mm
    t = design[2] * 1000
    s = design[3] * 1000
    
    from config import config
    W = config.operating.base_width * 1000  # mm
    t_base = config.operating.base_thickness * 1000  # mm
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Draw base
    base = Rectangle((0, 0), N*(t+s)-s, t_base, 
                     facecolor='gray', edgecolor='black', linewidth=2)
    ax.add_patch(base)
    
    # Draw fins
    for i in range(N):
        x_pos = i * (t + s)
        fin = Rectangle((x_pos, t_base), t, H,
                       facecolor='silver', edgecolor='black', linewidth=1.5)
        ax.add_patch(fin)
    
    # Add dimensions
    ax.annotate('', xy=(0, -5), xytext=(N*(t+s)-s, -5),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(N*(t+s)/2-s/2, -10, f'Total Width: {N*(t+s)-s:.1f} mm',
            ha='center', fontsize=11, fontweight='bold', color='red')
    
    ax.annotate('', xy=(-5, 0), xytext=(-5, t_base+H),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
    ax.text(-12, (t_base+H)/2, f'Height: {H:.1f} mm',
            ha='center', va='center', rotation=90,
            fontsize=11, fontweight='bold', color='blue')
    
    # Add specifications
    specs_text = (f"Design Specifications:\n"
                 f"Number of fins: {N}\n"
                 f"Fin height: {H:.1f} mm\n"
                 f"Fin thickness: {t:.2f} mm\n"
                 f"Fin spacing: {s:.2f} mm\n"
                 f"Air velocity: {design[4]:.1f} m/s")
    
    ax.text(0.98, 0.97, specs_text,
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
            fontsize=10, family='monospace')
    
    ax.set_xlim(-20, N*(t+s))
    ax.set_ylim(-20, H+t_base+10)
    ax.set_aspect('equal')
    ax.set_xlabel('Width [mm]', fontweight='bold')
    ax.set_ylabel('Height [mm]', fontweight='bold')
    ax.set_title('Heat Sink Geometry (Side View)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Geometry visualization saved to {save_path}")


def plot_sensitivity_analysis(sensitivity_data: Dict, save_path: str = 'sensitivity.png'):
    """
    Create tornado diagram for sensitivity analysis.
    
    Args:
        sensitivity_data: Dictionary with parameter sensitivities
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    params = list(sensitivity_data.keys())
    sensitivities = list(sensitivity_data.values())
    
    # Sort by absolute sensitivity
    sorted_indices = sorted(range(len(sensitivities)), 
                          key=lambda i: abs(sensitivities[i]), reverse=True)
    params_sorted = [params[i] for i in sorted_indices]
    sens_sorted = [sensitivities[i] for i in sorted_indices]
    
    y_pos = np.arange(len(params_sorted))
    colors = ['red' if s < 0 else 'green' for s in sens_sorted]
    
    ax.barh(y_pos, sens_sorted, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(params_sorted)
    ax.set_xlabel('Sensitivity [% change in objective]', fontweight='bold')
    ax.set_title('Sensitivity Analysis (Tornado Diagram)', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Sensitivity analysis plot saved to {save_path}")


def create_all_visualizations(optimized_design: np.ndarray, 
                              baseline_design: np.ndarray,
                              convergence_history: List[float],
                              comparison: Dict,
                              output_dir: str = './results/'):
    """
    Generate all visualization plots.
    
    Args:
        optimized_design: Optimal design vector
        baseline_design: Baseline design vector
        convergence_history: Optimization convergence data
        comparison: Comparison dictionary
        output_dir: Directory to save plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # 1. Convergence plot
    plot_convergence(convergence_history, 
                    save_path=os.path.join(output_dir, 'convergence.png'))
    
    # 2. Design comparison
    plot_design_comparison(comparison,
                          save_path=os.path.join(output_dir, 'design_comparison.png'))
    
    # 3. Geometry visualization
    plot_heat_sink_geometry(optimized_design,
                           save_path=os.path.join(output_dir, 'geometry_optimized.png'))
    plot_heat_sink_geometry(baseline_design,
                           save_path=os.path.join(output_dir, 'geometry_baseline.png'))
    
    print("\n" + "="*70)
    print(f"All visualizations saved to {output_dir}")
    print("="*70)


if __name__ == "__main__":
    # Test visualization module
    print("Testing visualization module...")
    
    from config import config
    
    # Test convergence plot
    test_convergence = [100 - i*0.5 for i in range(100)]
    plot_convergence(test_convergence, 'test_convergence.png')
    
    # Test geometry plot
    x_test = config.get_baseline_design()
    plot_heat_sink_geometry(x_test, 'test_geometry.png')
    
    print("\nTest plots generated successfully!")"""
Visualization Module for Heat Sink Optimization
Course: Transport Phenomena
Author: Sanskar Gunde (MM24B005)
Date: November 2025

This module creates all visualizations including:
- Convergence plots
- Parametric studies
- Performance comparisons
- Temperature distributions
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from matplotlib.patches import Rectangle

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_convergence(convergence_history: List[float], save_path: str = 'convergence.png'):
    """
    Plot optimization convergence history.
    
    Args:
        convergence_history: List of best objective values per iteration
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = list(range(1, len(convergence_history) + 1))
    ax.plot(iterations, convergence_history, 'b-', linewidth=2, label='Best Objective')
    
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Objective Function Value', fontsize=12, fontweight='bold')
    ax.set_title('Optimization Convergence History', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add improvement annotation
    improvement = (1 - convergence_history[-1]/convergence_history[0]) * 100
    ax.text(0.98, 0.95, f'Improvement: {improvement:.1f}%',
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Convergence plot saved to {save_path}")


def plot_design_comparison(comparison: Dict, save_path: str = 'design_comparison.png'):
    """
    Plot bar charts comparing different designs.
    
    Args:
        comparison: Dictionary from compare_designs or compare_with_literature
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    methods = list(comparison.keys())
    
    # Metric 1: Base Temperature
    temps = [comparison[m].get('T_base', 0) for m in methods]
    axes[0, 0].barh(methods, temps, color='orangered', alpha=0.7)
    axes[0, 0].set_xlabel('Base Temperature [°C]', fontweight='bold')
    axes[0, 0].set_title('Thermal Performance', fontweight='bold')
    axes[0, 0].axvline(x=85, color='red', linestyle='--', label='T_max limit')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Metric 2: Fan Power
    powers = [comparison[m].get('P_fan', comparison[m].get('fan_power', 0)) for m in methods]
    axes[0, 1].barh(methods, powers, color='steelblue', alpha=0.7)
    axes[0, 1].set_xlabel('Fan Power [W]', fontweight='bold')
    axes[0, 1].set_title('Power Consumption', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Metric 3: Material Cost
    costs = [comparison[m].get('material_cost', comparison[m].get('cost', 0)) for m in methods]
    axes[1, 0].barh(methods, costs, color='green', alpha=0.7)
    axes[1, 0].set_xlabel('Cost [$]', fontweight='bold')
    axes[1, 0].set_title('Material Cost', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Metric 4: PEC (Performance Evaluation Criterion)
    pecs = [comparison[m].get('pec', 0) for m in methods]
    axes[1, 1].barh(methods, pecs, color='purple', alpha=0.7)
    axes[1, 1].set_xlabel('PEC [W/(K·W)]', fontweight='bold')
    axes[1, 1].set_title('Performance Evaluation Criterion\n(Higher is Better)', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Design comparison plot saved to {save_path}")


def plot_parametric_study(param_name: str, param_values: np.ndarray,
                         results: Dict, save_path: str = 'parametric_study.png'):
    """
    Plot results of parametric study.
    
    Args:
        param_name: Name of parameter varied
        param_values: Array of parameter values
        results: Dictionary with arrays of results
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Temperature vs parameter
    axes[0, 0].plot(param_values, results['T_base'], 'o-', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel(param_name, fontweight='bold')
    axes[0, 0].set_ylabel('Base Temperature [°C]', fontweight='bold')
    axes[0, 0].set_title(f'Temperature vs {param_name}', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Fan power vs parameter
    axes[0, 1].plot(param_values, results['P_fan'], 'o-', linewidth=2, 
                   markersize=6, color='steelblue')
    axes[0, 1].set_xlabel(param_name, fontweight='bold')
    axes[0, 1].set_ylabel('Fan Power [W]', fontweight='bold')
    axes[0, 1].set_title(f'Fan Power vs {param_name}', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Thermal resistance vs parameter
    axes[1, 0].plot(param_values, results['R_th'], 'o-', linewidth=2,
                   markersize=6, color='red')
    axes[1, 0].set_xlabel(param_name, fontweight='bold')
    axes[1, 0].set_ylabel('Thermal Resistance [K/W]', fontweight='bold')
    axes[1, 0].set_title(f'Thermal Resistance vs {param_name}', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # PEC vs parameter
    axes[1, 1].plot(param_values, results['PEC'], 'o-', linewidth=2,
                   markersize=6, color='purple')
    axes[1, 1].set_xlabel(param_name, fontweight='bold')
    axes[1, 1].set_ylabel('PEC [W/(K·W)]', fontweight='bold')
    axes[1, 1].set_title(f'PEC vs {param_name}', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Parametric study plot saved to {save_path}")


def plot_pareto_frontier(designs: List[Tuple], save_path: str = 'pareto_frontier.png'):
    """
    Plot Pareto frontier for multi-objective optimization.
    
    Args:
        designs: List of (T_base, P_fan, cost) tuples
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(14, 10))
    
    # 2D projections
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    
    temps = [d[0] for d in designs]
    powers = [d[1] for d in designs]
    costs = [d[2] for d in designs]
    
    # Temperature vs Fan Power
    ax1.scatter(temps, powers, c='blue', s=50, alpha=0.6)
    ax1.set_xlabel('Base Temperature [°C]', fontweight='bold')
    ax1.set_ylabel('Fan Power [W]', fontweight='bold')
    ax1.set_title('Temperature vs Power Trade-off', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Temperature vs Cost
    ax2.scatter(temps, costs, c='green', s=50, alpha=0.6)
    ax2.set_xlabel('Base Temperature [°C]', fontweight='bold')
    ax2.set_ylabel('Cost [$]', fontweight='bold')
    ax2.set_title('Temperature vs Cost Trade-off', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Fan Power vs Cost
    ax3.scatter(powers, costs, c='red', s=50, alpha=0.6)
    ax3.set_xlabel('Fan Power [W]', fontweight='bold')
    ax3.set_ylabel('Cost [$]', fontweight='bold')
    ax3.set_title('Power vs Cost Trade-off', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 3D Pareto frontier
    ax4.scatter(temps, powers, costs, c='purple', s=50, alpha=0.6)
    ax4.set_xlabel('Temperature [°C]', fontweight='bold')
    ax4.set_ylabel('Fan Power [W]', fontweight='bold')
    ax4.set_zlabel('Cost [$]', fontweight='bold')
    ax4.set_title('3D Pareto Frontier', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Pareto frontier plot saved to {save_path}")


def plot_heat_sink_geometry(design: np.ndarray, save_path: str = 'geometry.png'):
    """
    Visualize heat sink geometry (2D schematic).
    
    Args:
        design: Design vector [N, H, t, s, v]
        save_path: Path to save figure
    """
    N = int(round(design[0]))
    H = design[1] * 1000  # Convert to mm
    t = design[2] * 1000
    s = design[3] * 1000
    
    from config import config
    W = config.operating.base_width * 1000  # mm
    t_base = config.operating.base_thickness * 1000  # mm
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Draw base
    base = Rectangle((0, 0), N*(t+s)-s, t_base, 
                     facecolor='gray', edgecolor='black', linewidth=2)
    ax.add_patch(base)
    
    # Draw fins
    for i in range(N):
        x_pos = i * (t + s)
        fin = Rectangle((x_pos, t_base), t, H,
                       facecolor='silver', edgecolor='black', linewidth=1.5)
        ax.add_patch(fin)
    
    # Add dimensions
    ax.annotate('', xy=(0, -5), xytext=(N*(t+s)-s, -5),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(N*(t+s)/2-s/2, -10, f'Total Width: {N*(t+s)-s:.1f} mm',
            ha='center', fontsize=11, fontweight='bold', color='red')
    
    ax.annotate('', xy=(-5, 0), xytext=(-5, t_base+H),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
    ax.text(-12, (t_base+H)/2, f'Height: {H:.1f} mm',
            ha='center', va='center', rotation=90,
            fontsize=11, fontweight='bold', color='blue')
    
    # Add specifications
    specs_text = (f"Design Specifications:\n"
                 f"Number of fins: {N}\n"
                 f"Fin height: {H:.1f} mm\n"
                 f"Fin thickness: {t:.2f} mm\n"
                 f"Fin spacing: {s:.2f} mm\n"
                 f"Air velocity: {design[4]:.1f} m/s")
    
    ax.text(0.98, 0.97, specs_text,
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
            fontsize=10, family='monospace')
    
    ax.set_xlim(-20, N*(t+s))
    ax.set_ylim(-20, H+t_base+10)
    ax.set_aspect('equal')
    ax.set_xlabel('Width [mm]', fontweight='bold')
    ax.set_ylabel('Height [mm]', fontweight='bold')
    ax.set_title('Heat Sink Geometry (Side View)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Geometry visualization saved to {save_path}")


def plot_sensitivity_analysis(sensitivity_data: Dict, save_path: str = 'sensitivity.png'):
    """
    Create tornado diagram for sensitivity analysis.
    
    Args:
        sensitivity_data: Dictionary with parameter sensitivities
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    params = list(sensitivity_data.keys())
    sensitivities = list(sensitivity_data.values())
    
    # Sort by absolute sensitivity
    sorted_indices = sorted(range(len(sensitivities)), 
                          key=lambda i: abs(sensitivities[i]), reverse=True)
    params_sorted = [params[i] for i in sorted_indices]
    sens_sorted = [sensitivities[i] for i in sorted_indices]
    
    y_pos = np.arange(len(params_sorted))
    colors = ['red' if s < 0 else 'green' for s in sens_sorted]
    
    ax.barh(y_pos, sens_sorted, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(params_sorted)
    ax.set_xlabel('Sensitivity [% change in objective]', fontweight='bold')
    ax.set_title('Sensitivity Analysis (Tornado Diagram)', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Sensitivity analysis plot saved to {save_path}")


def create_all_visualizations(optimized_design: np.ndarray, 
                              baseline_design: np.ndarray,
                              convergence_history: List[float],
                              comparison: Dict,
                              output_dir: str = './results/'):
    """
    Generate all visualization plots.
    
    Args:
        optimized_design: Optimal design vector
        baseline_design: Baseline design vector
        convergence_history: Optimization convergence data
        comparison: Comparison dictionary
        output_dir: Directory to save plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # 1. Convergence plot
    plot_convergence(convergence_history, 
                    save_path=os.path.join(output_dir, 'convergence.png'))
    
    # 2. Design comparison
    plot_design_comparison(comparison,
                          save_path=os.path.join(output_dir, 'design_comparison.png'))
    
    # 3. Geometry visualization
    plot_heat_sink_geometry(optimized_design,
                           save_path=os.path.join(output_dir, 'geometry_optimized.png'))
    plot_heat_sink_geometry(baseline_design,
                           save_path=os.path.join(output_dir, 'geometry_baseline.png'))
    
    print("\n" + "="*70)
    print(f"All visualizations saved to {output_dir}")
    print("="*70)


if __name__ == "__main__":
    # Test visualization module
    print("Testing visualization module...")
    
    from config import config
    
    # Test convergence plot
    test_convergence = [100 - i*0.5 for i in range(100)]
    plot_convergence(test_convergence, 'test_convergence.png')
    
    # Test geometry plot
    x_test = config.get_baseline_design()
    plot_heat_sink_geometry(x_test, 'test_geometry.png')
    
    print("\nTest plots generated successfully!")