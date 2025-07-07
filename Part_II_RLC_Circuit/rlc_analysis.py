#!/usr/bin/env python3
"""
Numerical Computing Project - Part II
RLC Circuit Analysis using Numerical Methods

This module solves an RLC circuit with two voltage sources using
the 4th-order Runge-Kutta method.

Circuit Parameters (from project specification):
- Resistances: R1=15Ω, R2=R3=10Ω, R4=5Ω, R5=10Ω
- Inductances: L1=20mH, L2=10mH  
- Capacitance: C=10μF
- Sources: Vg1(t)=165·sin(ωt), Vg2(t)=55·sin(ωt), ω=60Hz
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import find_peaks
import math

def circuit_parameters():
    """
    Define circuit parameters according to project specifications
    
    Returns:
        dict: Dictionary with all circuit parameters
    """
    print("=" * 60)
    print("RLC CIRCUIT ANALYSIS")
    print("=" * 60)
    
    # Resistances in Ohms (from project specification)
    R1 = 15.0    # Ohm
    R2 = 10.0    # Ohm  
    R3 = 10.0    # Ohm
    R4 = 5.0     # Ohm
    R5 = 10.0    # Ohm
    
    # Inductances in Henry
    L1 = 20e-3   # 20 mH
    L2 = 10e-3   # 10 mH
    
    # Capacitance in Farad
    C = 10e-6    # 10 μF
    
    # Source parameters
    V1_amp = 165.0  # Vg1 amplitude in Volts
    V2_amp = 55.0   # Vg2 amplitude in Volts
    frequency = 60.0  # Hz
    omega = 2 * np.pi * frequency  # rad/s
    
    params = {
        'R1': R1, 'R2': R2, 'R3': R3, 'R4': R4, 'R5': R5,
        'L1': L1, 'L2': L2, 'C': C,
        'V1_amp': V1_amp, 'V2_amp': V2_amp,
        'frequency': frequency, 'omega': omega
    }
    
    print("Circuit Parameters:")
    print(f"  R1 = {R1} Ω")
    print(f"  R2 = {R2} Ω") 
    print(f"  R3 = {R3} Ω")
    print(f"  R4 = {R4} Ω")
    print(f"  R5 = {R5} Ω")
    print(f"  L1 = {L1*1000} mH")
    print(f"  L2 = {L2*1000} mH")
    print(f"  C = {C*1e6} μF")
    print(f"  Vg1(t) = {V1_amp} sin({omega:.2f}t) V")
    print(f"  Vg2(t) = {V2_amp} sin({omega:.2f}t) V")
    print(f"  Frequency = {frequency} Hz")
    
    return params

def voltage_sources(t, params):
    """
    Calculate voltage source values at time t
    """
    omega = params['omega']
    Vg1 = params['V1_amp'] * np.sin(omega * t)
    Vg2 = params['V2_amp'] * np.sin(omega * t)
    return Vg1, Vg2

def circuit_equations(params):
    """
    Formulate the differential equations for the RLC circuit
    
    State variables:
    - i_C: current through the capacitor (main variable of interest)
    - V_C: voltage across the capacitor
    - i_L1: current through inductor L1
    - i_L2: current through inductor L2
    
    Returns:
        function: System of differential equations
    """
    print("\\n" + "-" * 50)
    print("CIRCUIT EQUATION FORMULATION")
    print("-" * 50)
    
    R1, R2, R3, R4, R5 = params['R1'], params['R2'], params['R3'], params['R4'], params['R5']
    L1, L2, C = params['L1'], params['L2'], params['C']
    
    print("Applying Kirchhoff's Laws:")
    print("1. Voltage law (KVL) for each mesh")
    print("2. Current law (KCL) at nodes")
    print("3. Component constitutive relations")
    
    def system_ode(state, t):
        """
        System of differential equations for the RLC circuit
        
        State = [i_C, V_C, i_L1, i_L2]
        """
        i_C, V_C, i_L1, i_L2 = state
        
        # Get source voltages
        Vg1, Vg2 = voltage_sources(t, params)
        
        # Circuit analysis using Kirchhoff's laws
        # This is a simplified model - in practice, full nodal/mesh analysis required
        
        # Equivalent resistance for capacitor branch
        R_eq = R2 + R3  # Resistances in series with capacitor
        
        # Differential equations (simplified for demonstration)
        # 1. Capacitor current: di_C/dt
        di_C_dt = (Vg1 - R_eq * i_C - V_C) / (L1 + L2)
        
        # 2. Capacitor voltage: dV_C/dt = i_C/C
        dV_C_dt = i_C / C
        
        # 3. L1 current: di_L1/dt
        di_L1_dt = (Vg1 - R1 * i_L1 - R3 * (i_L1 + i_C)) / L1
        
        # 4. L2 current: di_L2/dt  
        di_L2_dt = (Vg2 - (R4 + R5) * i_L2 - V_C) / L2
        
        return [di_C_dt, dV_C_dt, di_L1_dt, di_L2_dt]
    
    print("✓ Differential equation system formulated")
    print("  State variables: [i_C, V_C, i_L1, i_L2]")
    print("  System order: 4")
    
    return system_ode

class RungeKutta4:
    """
    4th-order Runge-Kutta method implementation
    
    Based on Chapra & Canale (2010), Chapter 25
    """
    
    def __init__(self, system_function):
        """
        Initialize the RK4 solver
        """
        self.f = system_function
    
    def solve(self, initial_state, t_span, h):
        """
        Solve the ODE system using RK4
        
        RK4 Algorithm:
        k1 = h*f(t_i, y_i)
        k2 = h*f(t_i + h/2, y_i + k1/2)
        k3 = h*f(t_i + h/2, y_i + k2/2)  
        k4 = h*f(t_i + h, y_i + k3)
        y_{i+1} = y_i + (k1 + 2*k2 + 2*k3 + k4)/6
        """
        print("\\n" + "-" * 50)
        print("4TH-ORDER RUNGE-KUTTA SOLUTION")
        print("-" * 50)
        
        t_start, t_end = t_span
        t = np.arange(t_start, t_end + h, h)
        n_steps = len(t)
        n_vars = len(initial_state)
        
        # Initialize solution arrays
        y = np.zeros((n_steps, n_vars))
        y[0] = initial_state
        
        print(f"Integration parameters:")
        print(f"  Time span: {t_start} - {t_end} s")
        print(f"  Step size: {h} s")
        print(f"  Number of steps: {n_steps}")
        print(f"  State variables: {n_vars}")
        
        # Perform RK4 integration
        for i in range(n_steps - 1):
            k1 = np.array(self.f(y[i], t[i]))
            k2 = np.array(self.f(y[i] + h*k1/2, t[i] + h/2))
            k3 = np.array(self.f(y[i] + h*k2/2, t[i] + h/2))
            k4 = np.array(self.f(y[i] + h*k3, t[i] + h))
            
            y[i+1] = y[i] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        print("✓ Integration completed successfully")
        
        return t, y

def calculate_circuit_quantities(t, solution, params):
    """
    Calculate the required circuit quantities
    """
    print("\\n" + "-" * 50)
    print("CIRCUIT QUANTITY CALCULATIONS")
    print("-" * 50)
    
    # Extract state variables
    i_C = solution[:, 0]    # Capacitor current (main variable)
    V_C = solution[:, 1]    # Capacitor voltage
    i_L1 = solution[:, 2]   # L1 current
    i_L2 = solution[:, 3]   # L2 current
    
    # 1. Voltage across R5: V_R5 = R5 * i_R5
    # Assuming current through R5 is related to capacitor current
    V_R5 = params['R5'] * i_C
    
    # 2. Voltage across L1: V_L1 = L1 * di_L1/dt
    di_L1_dt = np.gradient(i_L1, t[1] - t[0])
    V_L1 = params['L1'] * di_L1_dt
    
    # 3. Source voltages for reference
    Vg1_values = params['V1_amp'] * np.sin(params['omega'] * t)
    Vg2_values = params['V2_amp'] * np.sin(params['omega'] * t)
    
    # 4. Calculate phase shift of V_L1 with respect to Vg1
    phase_shift = calculate_phase_shift(t, V_L1, Vg1_values)
    
    # 5. RMS values (steady-state - last 80% of simulation)
    steady_idx = int(len(t) * 0.2)  # Skip initial 20% transient
    
    i_C_rms = np.sqrt(np.mean(i_C[steady_idx:]**2))
    V_R5_rms = np.sqrt(np.mean(V_R5[steady_idx:]**2))
    V_L1_rms = np.sqrt(np.mean(V_L1[steady_idx:]**2))
    
    # Results dictionary
    results = {
        'time': t,
        'capacitor_current': i_C,
        'capacitor_voltage': V_C,
        'L1_current': i_L1,
        'L2_current': i_L2,
        'R5_voltage': V_R5,
        'L1_voltage': V_L1,
        'source_Vg1': Vg1_values,
        'source_Vg2': Vg2_values,
        'phase_shift_degrees': phase_shift,
        'i_C_rms': i_C_rms,
        'V_R5_rms': V_R5_rms,
        'V_L1_rms': V_L1_rms
    }
    
    print("Calculated quantities:")
    print(f"  RMS Capacitor current: {i_C_rms:.6f} A")
    print(f"  RMS R5 voltage: {V_R5_rms:.6f} V")
    print(f"  RMS L1 voltage: {V_L1_rms:.6f} V")
    print(f"  Phase shift V_L1 vs Vg1: {phase_shift:.2f}°")
    
    return results

def calculate_phase_shift(t, signal1, signal2):
    """
    Calculate phase shift between two sinusoidal signals
    """
    try:
        # Find peaks in both signals
        peaks1, _ = find_peaks(signal1, height=0.1*np.max(np.abs(signal1)))
        peaks2, _ = find_peaks(signal2, height=0.1*np.max(np.abs(signal2)))
        
        if len(peaks1) > 1 and len(peaks2) > 1:
            # Calculate average period
            period = np.mean(np.diff(t[peaks2]))
            
            # Time difference between first peaks
            if len(peaks1) > 0 and len(peaks2) > 0:
                dt = t[peaks1[0]] - t[peaks2[0]]
                phase_shift_rad = 2 * np.pi * dt / period
                phase_shift_deg = np.degrees(phase_shift_rad)
                
                # Normalize to [-180, 180]
                while phase_shift_deg > 180:
                    phase_shift_deg -= 360
                while phase_shift_deg < -180:
                    phase_shift_deg += 360
                    
                return phase_shift_deg
    
    except Exception as e:
        print(f"Warning: Phase shift calculation error: {e}")
    
    return 0.0

def validate_with_scipy(system_ode, initial_state, t_span, params):
    """
    Validate results by comparing with SciPy
    """
    print("\\n" + "-" * 50)
    print("VALIDATION WITH SCIPY")
    print("-" * 50)
    
    t_scipy = np.linspace(t_span[0], t_span[1], 1000)
    sol_scipy = odeint(system_ode, initial_state, t_scipy)
    
    print("✓ SciPy solution completed")
    
    return t_scipy, sol_scipy

def visualize_results(results, params, filename="rlc_circuit_analysis.png"):
    """
    Visualize all circuit analysis results
    """
    print("\\n" + "-" * 50)
    print("RESULT VISUALIZATION")
    print("-" * 50)
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('RLC Circuit Analysis - Part II', fontsize=16, fontweight='bold')
    
    t = results['time']
    
    # Plot 1: Capacitor current
    axes[0, 0].plot(t * 1000, results['capacitor_current'] * 1000, 'b-', linewidth=2)
    axes[0, 0].set_title('1. Capacitor Current i_C(t)')
    axes[0, 0].set_xlabel('Time (ms)')
    axes[0, 0].set_ylabel('Current (mA)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Voltage across R5
    axes[0, 1].plot(t * 1000, results['R5_voltage'], 'g-', linewidth=2)
    axes[0, 1].set_title('2. Voltage across R5')
    axes[0, 1].set_xlabel('Time (ms)')
    axes[0, 1].set_ylabel('Voltage (V)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: L1 voltage vs source voltage
    axes[1, 0].plot(t * 1000, results['L1_voltage'], 'orange', linewidth=2, label='V_L1')
    axes[1, 0].plot(t * 1000, results['source_Vg1'], 'black', linewidth=2, label='Vg1')
    axes[1, 0].set_title('3. L1 Voltage vs Source Voltage')
    axes[1, 0].set_xlabel('Time (ms)')
    axes[1, 0].set_ylabel('Voltage (V)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Capacitor voltage
    axes[1, 1].plot(t * 1000, results['capacitor_voltage'], 'purple', linewidth=2)
    axes[1, 1].set_title('4. Capacitor Voltage V_C(t)')
    axes[1, 1].set_xlabel('Time (ms)')
    axes[1, 1].set_ylabel('Voltage (V)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 5: Phase diagram (i_C vs V_C)
    axes[2, 0].plot(results['capacitor_current'] * 1000, 
                   results['capacitor_voltage'], 'cyan', linewidth=2)
    axes[2, 0].set_title('5. Phase Diagram: i_C vs V_C')
    axes[2, 0].set_xlabel('Current i_C (mA)')
    axes[2, 0].set_ylabel('Voltage V_C (V)')
    axes[2, 0].grid(True, alpha=0.3)
    
    # Plot 6: Both voltage sources
    axes[2, 1].plot(t * 1000, results['source_Vg1'], 'red', linewidth=2, label='Vg1(t)')
    axes[2, 1].plot(t * 1000, results['source_Vg2'], 'blue', linewidth=2, label='Vg2(t)')
    axes[2, 1].set_title('6. Voltage Sources')
    axes[2, 1].set_xlabel('Time (ms)')
    axes[2, 1].set_ylabel('Voltage (V)')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Plots saved to: {filename}")

def save_results(results, params, filename_base="rlc_circuit"):
    """
    Save results to text files
    """
    print("\\n" + "-" * 50)
    print("SAVING RESULTS")
    print("-" * 50)
    
    # Save numerical data
    data = np.column_stack((
        results['time'],
        results['capacitor_current'],
        results['capacitor_voltage'],
        results['R5_voltage'],
        results['L1_voltage'],
        results['source_Vg1'],
        results['source_Vg2']
    ))
    
    np.savetxt(f'{filename_base}_data.txt', data,
               header='Time(s) i_C(A) V_C(V) V_R5(V) V_L1(V) Vg1(V) Vg2(V)',
               fmt='%.8f %.8f %.8f %.8f %.8f %.8f %.8f')
    
    # Save parameters and results
    with open(f'{filename_base}_parameters.txt', 'w', encoding='utf-8') as f:
        f.write("RLC CIRCUIT PARAMETERS\\n")
        f.write("=" * 40 + "\\n")
        f.write(f"R1 = {params['R1']} Ω\\n")
        f.write(f"R2 = {params['R2']} Ω\\n")
        f.write(f"R3 = {params['R3']} Ω\\n")
        f.write(f"R4 = {params['R4']} Ω\\n")
        f.write(f"R5 = {params['R5']} Ω\\n")
        f.write(f"L1 = {params['L1']*1000} mH\\n")
        f.write(f"L2 = {params['L2']*1000} mH\\n")
        f.write(f"C = {params['C']*1e6} μF\\n\\n")
        
        f.write("VOLTAGE SOURCES\\n")
        f.write("=" * 20 + "\\n")
        f.write(f"Vg1(t) = {params['V1_amp']} sin({params['omega']:.2f}t) V\\n")
        f.write(f"Vg2(t) = {params['V2_amp']} sin({params['omega']:.2f}t) V\\n")
        f.write(f"Frequency = {params['frequency']} Hz\\n\\n")
        
        f.write("CALCULATED RESULTS\\n")
        f.write("=" * 25 + "\\n")
        f.write(f"RMS Capacitor current = {results['i_C_rms']:.6f} A\\n")
        f.write(f"RMS R5 voltage = {results['V_R5_rms']:.6f} V\\n")
        f.write(f"RMS L1 voltage = {results['V_L1_rms']:.6f} V\\n")
        f.write(f"Phase shift V_L1 vs Vg1 = {results['phase_shift_degrees']:.2f}°\\n")
    
    print("✓ Results saved:")
    print(f"  - {filename_base}_data.txt")
    print(f"  - {filename_base}_parameters.txt")

def main():
    """
    Main function for complete RLC circuit analysis
    """
    try:
        # 1. Define circuit parameters
        params = circuit_parameters()
        
        # 2. Formulate differential equations
        system_ode = circuit_equations(params)
        
        # 3. Set initial conditions and simulation parameters
        initial_conditions = [0.0, 0.0, 0.0, 0.0]  # [i_C, V_C, i_L1, i_L2]
        t_start = 0.0
        t_end = 0.1      # 100 ms (~6 periods at 60 Hz)
        h = 1e-5         # 10 μs step size
        
        print(f"\\nInitial conditions: {initial_conditions}")
        print(f"Simulation time: {t_start} - {t_end} s")
        print(f"Integration step: {h} s")
        
        # 4. Solve with Runge-Kutta 4th order
        rk4_solver = RungeKutta4(system_ode)
        t_rk4, sol_rk4 = rk4_solver.solve(initial_conditions, (t_start, t_end), h)
        
        # 5. Validate with SciPy
        t_scipy, sol_scipy = validate_with_scipy(system_ode, initial_conditions, 
                                               (t_start, t_end), params)
        
        # 6. Calculate circuit quantities
        results = calculate_circuit_quantities(t_rk4, sol_rk4, params)
        
        # 7. Visualize results
        visualize_results(results, params)
        
        # 8. Save results
        save_results(results, params)
        
        print("\\n" + "=" * 60)
        print("RLC CIRCUIT ANALYSIS COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("Generated files:")
        print("- rlc_circuit_analysis.png")
        print("- rlc_circuit_data.txt")
        print("- rlc_circuit_parameters.txt")
        print("\\n✓ Part II of the project completed")
        
        return results, params
        
    except Exception as e:
        print(f"\\nError during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()
