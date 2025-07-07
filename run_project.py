"""
Numerical Computing Project - Complete Execution
Simple script to run both parts of the project
"""

import os
import sys
import subprocess

def get_python_command():
    """
    Detect which Python command is available on the system
    """
    commands = ['py', 'python', 'python3']
    
    for cmd in commands:
        try:
            # Try to run the command with --version to check if it exists
            result = subprocess.run([cmd, '--version'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            if result.returncode == 0:
                # Check if the output contains actual Python version info
                # and not Microsoft Store redirect message
                if 'Python' in result.stdout and 'Microsoft Store' not in result.stderr:
                    return cmd
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            continue
    
    # If no Python command found, default to 'py' (Windows Python Launcher)
    return 'py'

def main():
    print("=" * 60)
    print("NUMERICAL COMPUTING PROJECT")
    print("=" * 60)
    print("Part I: Image Splines")
    print("Part II: RLC Circuit Analysis")
    print()
    
    # Detect the appropriate Python command
    python_cmd = get_python_command()
    print(f"Using Python command: {python_cmd}")
    print()
    
    # Check if we need to run Part I first
    if not os.path.exists('bordes_panda_canny.jpg'):
        print("Running Part I - Step 1: Image Processing...")
        os.chdir("Part_I_Image_Splines")
        os.system(f"{python_cmd} image_processing.py")
        os.chdir("..")
        print("✓ Image processing completed")
    
    print("\\nRunning Part I - Step 2: Cubic Splines...")
    os.chdir("Part_I_Image_Splines")
    os.system(f"{python_cmd} cubic_splines.py")
    os.chdir("..")
    print("✓ Cubic splines completed")
    
    print("\\nRunning Part II: RLC Circuit Analysis...")
    os.chdir("Part_II_RLC_Circuit")
    os.system(f"{python_cmd} rlc_analysis.py")
    os.chdir("..")
    print("✓ RLC analysis completed")
    
    print("\\n" + "=" * 60)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Check the generated files:")
    print("- Part_I_Image_Splines/: Spline interpolation")
    print("- Part_II_RLC_Circuit/: Circuit analysis")
    print("- Documentacion/: Spanish documentation")

if __name__ == "__main__":
    main()
