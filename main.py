"""
Main Execution Module for Heat Sink Optimization
Course: Transport Phenomena
Author: Sanskar Gunde (MM24B005)
Date: November 2025

This is the main entry point for the heat sink optimization project.
It orchestrates the entire workflow:
1. Configuration and setup
2. Baseline design evaluation
3. Optimization using Differential Evolution
4. Performance comparison
5. Visualization generation
6. Results reporting
"""

import numpy as np
import os
import sys
from datetime import datetime

# Import project modules
from config import config
from heat_transfer import solve_base_temperature
from optimization import optimize_heat_sink, evaluate_design
from performance import compare_designs
from comparison import compare_with_literature
from visualization import create_all_visualizations
from utils import create_results_directory, save_results_to_file, print_banner


def main():
    """
    Main execution function for the heat sink optimization project.
    """
    # Print project banner
    print_banner()
    
    # Display configuration
    config.print_configuration()
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = create_results_directory(timestamp)
    
    print("\n" + "="*70)
    print("STEP 1: BASELINE DESIGN EVALUATION")
    print("="*70)
    
    # Evaluate baseline design
    x_baseline = config.get_baseline_design()
    print("\nEvaluating baseline design...")
    metrics_baseline = evaluate_design(x_baseline, verbose=True)
    
    print("\n" + "="*70)
    print("STEP 2: OPTIMIZATION")
    print("="*70)
    
    # Run optimization
    print("\nRunning Differential Evolution optimization...")
    x_optimized, J_opt, result_dict = optimize_heat_sink(verbose=True)
    
    # Evaluate optimized design
    print("\n\nEvaluating optimized design...")
    metrics_optimized = evaluate_design(x_optimized, verbose=True)
    
    print("\n" + "="*70)
    print("STEP 3: PERFORMANCE COMPARISON")
    print("="*70)
    
    # Compare baseline vs optimized
    designs_to_compare = {
        'Baseline': x_baseline,
        'Optimized': x_optimized
    }
    comparison_basic = compare_designs(designs_to_compare, verbose=True)
    
    # Compare with literature and other cooling methods
    print("\n")
    comparison_full = compare_with_literature(x_optimized, verbose=True)
    
    print("\n" + "="*70)
    print("STEP 4: RESULTS ANALYSIS")
    print("="*70)
    
    # Calculate improvements
    T_improvement = metrics_baseline['T_base'] - metrics_optimized['T_base']
    P_improvement = metrics_baseline['fan_power'] - metrics_optimized['fan_power']
    cost_improvement = metrics_baseline['material_cost'] - metrics_optimized['material_cost']
    
    print(f"\nIMPROVEMENTS ACHIEVED:")
    print(f"  Temperature reduction: {T_improvement:.2f}°C "
          f"({T_improvement/metrics_baseline['T_base']*100:.1f}%)")
    print(f"  Fan power reduction: {P_improvement:.3f}W "
          f"({P_improvement/metrics_baseline['fan_power']*100:.1f}%)")
    print(f"  Material cost reduction: ${cost_improvement:.2f} "
          f"({cost_improvement/metrics_baseline['material_cost']*100:.1f}%)")
    
    # Energy savings calculation
    hours_per_year = 365 * 24
    energy_savings_kwh = P_improvement * hours_per_year / 1000
    cost_per_kwh = 0.12  # $/kWh (typical electricity cost)
    annual_savings = energy_savings_kwh * cost_per_kwh
    
    print(f"\nENERGY AND COST SAVINGS (per heat sink per year):")
    print(f"  Energy saved: {energy_savings_kwh:.1f} kWh/year")
    print(f"  Cost saved: ${annual_savings:.2f}/year")
    print(f"  Payback period: {(metrics_optimized['material_cost']-metrics_baseline['material_cost'])/annual_savings:.2f} years")
    
    # Data center scale impact
    servers_per_datacenter = 50000  # Typical large data center
    total_annual_savings = annual_savings * servers_per_datacenter
    
    print(f"\nDATA CENTER SCALE IMPACT ({servers_per_datacenter:,} servers):")
    print(f"  Total energy saved: {energy_savings_kwh*servers_per_datacenter/1e6:.2f} GWh/year")
    print(f"  Total cost saved: ${total_annual_savings/1e6:.2f}M/year")
    print(f"  CO2 reduction: {energy_savings_kwh*servers_per_datacenter*0.5/1e6:.2f} kton/year")
    print(f"  (assuming 0.5 kg CO2/kWh)")
    
    print("\n" + "="*70)
    print("STEP 5: VISUALIZATION")
    print("="*70)
    
    # Generate all visualizations
    create_all_visualizations(
        optimized_design=x_optimized,
        baseline_design=x_baseline,
        convergence_history=result_dict['convergence_history'],
        comparison=comparison_full,
        output_dir=results_dir
    )
    
    print("\n" + "="*70)
    print("STEP 6: SAVING RESULTS")
    print("="*70)
    
    # Save detailed results to file
    results_data = {
        'timestamp': timestamp,
        'baseline_design': x_baseline,
        'optimized_design': x_optimized,
        'baseline_metrics': metrics_baseline,
        'optimized_metrics': metrics_optimized,
        'optimization_result': result_dict,
        'comparison': comparison_full,
        'improvements': {
            'temperature': T_improvement,
            'power': P_improvement,
            'cost': cost_improvement,
            'annual_energy_savings_kwh': energy_savings_kwh,
            'annual_cost_savings': annual_savings
        }
    }
    
    results_file = save_results_to_file(results_data, results_dir)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Plots saved to: {results_dir}")
    
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE!")
    print("="*70)
    
    # Final summary
    print(f"\nFINAL OPTIMIZED DESIGN:")
    print(f"  Number of fins: {int(round(x_optimized[0]))}")
    print(f"  Fin height: {x_optimized[1]*1000:.2f} mm")
    print(f"  Fin thickness: {x_optimized[2]*1000:.2f} mm")
    print(f"  Fin spacing: {x_optimized[3]*1000:.2f} mm")
    print(f"  Air velocity: {x_optimized[4]:.2f} m/s")
    print(f"\nPERFORMANCE:")
    print(f"  Base temperature: {metrics_optimized['T_base']:.2f}°C")
    print(f"  Thermal resistance: {metrics_optimized['thermal_resistance']:.4f} K/W")
    print(f"  Fan power: {metrics_optimized['fan_power']:.3f} W")
    print(f"  Material cost: ${metrics_optimized['material_cost']:.2f}")
    print(f"  PEC: {metrics_optimized['pec']:.2f}")
    
    print("\n" + "="*70)
    print("Thank you for using the Heat Sink Optimizer!")
    print("Author: Sanskar Gunde (MM24B005)")
    print("Course: Transport Phenomena")
    print("="*70 + "\n")
    
    return results_data


if __name__ == "__main__":
    try:
        # Run main optimization workflow
        results = main()
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n\nERROR: An unexpected error occurred:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)