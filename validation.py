"""
Validation and Unit Testing Module
Course: Transport Phenomena
Author: Sanskar Gunde (MM24B005)
Date: November 2025

This module contains unit tests and validation checks for all
major functions in the heat sink optimization project.
"""

import numpy as np
import sys

from config import config
from heat_transfer import (calculate_fin_efficiency, solve_base_temperature,
                           calculate_thermal_resistance)
from fluid_dynamics import (calculate_reynolds_number, calculate_nusselt_number,
                            calculate_convection_coefficient, calculate_pressure_drop)
from optimization import objective_function, evaluate_design
from performance import calculate_material_mass, calculate_all_metrics


class ValidationTests:
    """
    Comprehensive validation tests for the heat sink optimization project.
    """
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []
    
    def run_test(self, test_name: str, test_func, expected_result=None, 
                 tolerance=1e-6):
        """
        Run a single test and record result.
        
        Args:
            test_name: Name of the test
            test_func: Function that returns (actual, expected) or (success, message)
            expected_result: Expected result (optional)
            tolerance: Tolerance for numerical comparison
        """
        print(f"\nRunning: {test_name}")
        print("-" * 70)
        
        try:
            result = test_func()
            
            if isinstance(result, tuple) and len(result) == 2:
                if isinstance(result[0], bool):
                    # Boolean test result
                    success, message = result
                    if success:
                        print(f"✓ PASSED: {message}")
                        self.passed += 1
                        self.tests.append((test_name, True, message))
                    else:
                        print(f"✗ FAILED: {message}")
                        self.failed += 1
                        self.tests.append((test_name, False, message))
                else:
                    # Numerical comparison
                    actual, expected = result
                    if abs(actual - expected) < tolerance:
                        print(f"✓ PASSED: actual={actual:.6f}, expected={expected:.6f}")
                        self.passed += 1
                        self.tests.append((test_name, True, f"Value: {actual:.6f}"))
                    else:
                        print(f"✗ FAILED: actual={actual:.6f}, expected={expected:.6f}, "
                              f"diff={abs(actual-expected):.6e}")
                        self.failed += 1
                        self.tests.append((test_name, False, 
                                         f"Mismatch: {actual:.6f} vs {expected:.6f}"))
            else:
                print(f"✓ PASSED: {result}")
                self.passed += 1
                self.tests.append((test_name, True, str(result)))
                
        except Exception as e:
            print(f"✗ FAILED with exception: {e}")
            self.failed += 1
            self.tests.append((test_name, False, f"Exception: {e}"))
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*70)
        print("VALIDATION TEST SUMMARY")
        print("="*70)
        print(f"Total tests: {self.passed + self.failed}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        
        if self.failed > 0:
            print("\nFailed tests:")
            for name, passed, message in self.tests:
                if not passed:
                    print(f"  - {name}: {message}")
        
        success_rate = self.passed / (self.passed + self.failed) * 100
        print(f"\nSuccess rate: {success_rate:.1f}%")
        print("="*70)
        
        return self.failed == 0


def test_fin_efficiency():
    """Test fin efficiency calculation against known values."""
    
    def test():
        # Test case: Short, thick fin should have high efficiency
        H = 0.02  # 20 mm
        t = 0.003  # 3 mm
        h = 50.0  # W/m²·K
        k = 205.0  # Aluminum
        
        eta = calculate_fin_efficiency(H, t, h, k)
        
        # Short fins should have efficiency close to 1
        if eta > 0.95:
            return (True, f"Fin efficiency = {eta:.4f} (expected > 0.95)")
        else:
            return (False, f"Fin efficiency = {eta:.4f} too low for short fin")
    
    return test


def test_reynolds_number():
    """Test Reynolds number calculation."""
    
    def test():
        v = 5.0  # m/s
        D_h = 0.008  # m (8 mm)
        T_film = 25.0  # °C
        
        Re = calculate_reynolds_number(v, D_h, T_film)
        
        # Expected Re = v*D_h/nu ≈ 5*0.008/1.562e-5 ≈ 2562
        expected = 2562
        
        if abs(Re - expected) / expected < 0.05:  # 5% tolerance
            return (True, f"Re = {Re:.0f} (expected ≈ {expected})")
        else:
            return (False, f"Re = {Re:.0f} differs significantly from {expected}")
    
    return test


def test_nusselt_number_laminar():
    """Test Nusselt number for laminar flow."""
    
    def test():
        Re = 1000  # Laminar
        Pr = 0.707
        L = 0.05
        D_h = 0.008
        
        Nu = calculate_nusselt_number(Re, Pr, L, D_h)
        
        # For laminar flow, Nu should be around 7.54
        expected = 7.54
        
        if abs(Nu - expected) / expected < 0.3:  # 30% tolerance (entry effects)
            return (True, f"Nu = {Nu:.2f} for laminar flow (expected ≈ {expected})")
        else:
            return (False, f"Nu = {Nu:.2f} unexpected for laminar flow")
    
    return test


def test_nusselt_number_turbulent():
    """Test Nusselt number for turbulent flow."""
    
    def test():
        Re = 5000  # Turbulent
        Pr = 0.707
        L = 0.05
        D_h = 0.008
        
        Nu = calculate_nusselt_number(Re, Pr, L, D_h)
        
        # Dittus-Boelter: Nu = 0.023 * Re^0.8 * Pr^0.4
        expected = 0.023 * (Re**0.8) * (Pr**0.4)
        
        if abs(Nu - expected) / expected < 0.2:  # 20% tolerance
            return (True, f"Nu = {Nu:.2f} for turbulent flow (expected ≈ {expected:.2f})")
        else:
            return (False, f"Nu = {Nu:.2f} differs from Dittus-Boelter: {expected:.2f}")
    
    return test


def test_material_mass():
    """Test material mass calculation."""
    
    def test():
        N = 20
        H = 0.05  # 50 mm
        t = 0.002  # 2 mm
        
        mass = calculate_material_mass(N, H, t)
        
        # Rough estimate: 20 fins × 50mm × 50mm × 2mm × 2700 kg/m³
        # Plus base: 50mm × 50mm × 3mm × 2700 kg/m³
        W = config.operating.base_width
        t_base = config.operating.base_thickness
        
        V_fins = N * W * H * t
        V_base = W * W * t_base
        V_total = V_fins + V_base
        expected = V_total * config.materials.aluminium_density
        
        if abs(mass - expected) / expected < 0.01:  # 1% tolerance
            return (True, f"Mass = {mass*1000:.1f}g (calculated correctly)")
        else:
            return (False, f"Mass calculation error: {mass:.6f} vs {expected:.6f}")
    
    return test


def test_temperature_solver():
    """Test temperature solver convergence."""
    
    def test():
        N = 20
        H = 0.04
        t = 0.002
        s = 0.005
        v = 3.0
        Q_load = 100.0
        T_inf = 25.0
        
        try:
            T_base, info = solve_base_temperature(N, H, t, s, v, Q_load, T_inf,
                                                 tol=1.0, max_iter=100)
            
            if info['converged']:
                # Check if temperature is reasonable
                if T_inf < T_base < T_inf + 100:
                    return (True, f"T_base = {T_base:.2f}°C (converged in "
                           f"{info['iterations']} iterations)")
                else:
                    return (False, f"T_base = {T_base:.2f}°C is unreasonable")
            else:
                return (False, f"Temperature solver did not converge")
                
        except Exception as e:
            return (False, f"Exception in temperature solver: {e}")
    
    return test


def test_constraint_validation():
    """Test design constraint validation."""
    
    def test():
        # Valid design
        x_valid = np.array([20, 0.04, 0.002, 0.005, 3.0])
        is_valid, msg = config.validate_design(x_valid)
        
        if not is_valid:
            return (False, f"Valid design rejected: {msg}")
        
        # Invalid design (spacing too small)
        x_invalid = np.array([20, 0.04, 0.002, 0.003, 3.0])  # s < 2*t
        is_valid, msg = config.validate_design(x_invalid)
        
        if is_valid:
            return (False, "Invalid design (s < 2t) was accepted")
        
        return (True, "Constraint validation working correctly")
    
    return test


def test_objective_function():
    """Test objective function evaluation."""
    
    def test():
        x = config.get_baseline_design()
        
        try:
            J = objective_function(x, verbose=False)
            
            if np.isfinite(J) and J > 0:
                return (True, f"Objective function = {J:.4f} (valid)")
            else:
                return (False, f"Objective function = {J} is invalid")
                
        except Exception as e:
            return (False, f"Exception in objective function: {e}")
    
    return test


def test_complete_evaluation():
    """Test complete design evaluation pipeline."""
    
    def test():
        x = config.get_baseline_design()
        
        try:
            metrics = evaluate_design(x, verbose=False)
            
            # Check if all expected metrics are present
            required_metrics = ['T_base', 'fan_power', 'thermal_resistance',
                              'total_mass', 'material_cost']
            
            for metric in required_metrics:
                if metric not in metrics:
                    return (False, f"Missing metric: {metric}")
            
            # Check if values are reasonable
            if not (25 < metrics['T_base'] < 150):
                return (False, f"T_base = {metrics['T_base']}°C is unreasonable")
            
            if not (0 < metrics['fan_power'] < 50):
                return (False, f"Fan power = {metrics['fan_power']}W is unreasonable")
            
            return (True, "Complete evaluation pipeline working")
            
        except Exception as e:
            return (False, f"Exception in evaluation: {e}")
    
    return test


def test_physical_consistency():
    """Test physical consistency of results."""
    
    def test():
        # Higher velocity should increase convection coefficient
        x1 = np.array([20, 0.04, 0.002, 0.005, 2.0])  # Low velocity
        x2 = np.array([20, 0.04, 0.002, 0.005, 6.0])  # High velocity
        
        Q_load = config.operating.Q_cpu
        T_inf = config.operating.T_ambient
        
        try:
            T1, info1 = solve_base_temperature(
                int(x1[0]), x1[1], x1[2], x1[3], x1[4], Q_load, T_inf, tol=1.0)
            T2, info2 = solve_base_temperature(
                int(x2[0]), x2[1], x2[2], x2[3], x2[4], Q_load, T_inf, tol=1.0)
            
            h1 = info1['h']
            h2 = info2['h']
            
            # Higher velocity should give higher h and lower T_base
            if h2 > h1 and T2 < T1:
                return (True, f"Physical consistency OK: h1={h1:.1f}, h2={h2:.1f}, "
                       f"T1={T1:.1f}°C, T2={T2:.1f}°C")
            else:
                return (False, f"Physical inconsistency: h1={h1:.1f}, h2={h2:.1f}, "
                       f"T1={T1:.1f}°C, T2={T2:.1f}°C")
                
        except Exception as e:
            return (False, f"Exception: {e}")
    
    return test


def run_all_validation_tests():
    """
    Run all validation tests and return success status.
    """
    validator = ValidationTests()
    
    print("\n" + "="*70)
    print("HEAT SINK OPTIMIZATION - VALIDATION TEST SUITE")
    print("="*70)
    
    # Run all tests
    validator.run_test("1. Fin Efficiency Calculation", test_fin_efficiency())
    validator.run_test("2. Reynolds Number Calculation", test_reynolds_number())
    validator.run_test("3. Nusselt Number (Laminar)", test_nusselt_number_laminar())
    validator.run_test("4. Nusselt Number (Turbulent)", test_nusselt_number_turbulent())
    validator.run_test("5. Material Mass Calculation", test_material_mass())
    validator.run_test("6. Temperature Solver Convergence", test_temperature_solver())
    validator.run_test("7. Constraint Validation", test_constraint_validation())
    validator.run_test("8. Objective Function", test_objective_function())
    validator.run_test("9. Complete Evaluation Pipeline", test_complete_evaluation())
    validator.run_test("10. Physical Consistency", test_physical_consistency())
    
    # Print summary
    all_passed = validator.print_summary()
    
    return all_passed


if __name__ == "__main__":
    # Run validation tests
    success = run_all_validation_tests()
    
    if success:
        print("\n✓ All validation tests passed! System is ready for optimization.")
        sys.exit(0)
    else:
        print("\n✗ Some validation tests failed. Please review the errors above.")
        sys.exit(1)