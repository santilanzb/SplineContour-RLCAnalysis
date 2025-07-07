#!/usr/bin/env python3
"""
Numerical Computing Project - Complete Execution
Simple script to run both parts of the project
"""

import os
import sys

def main():
    print("=" * 60)
    print("NUMERICAL COMPUTING PROJECT")
    print("=" * 60)
    print("Part I: Image Splines (Ada's work)")
    print("Part II: RLC Circuit Analysis")
    print()
    
    # Check if we need to run Part I first
    if not os.path.exists('bordes_panda_canny.jpg'):
        print("Running Part I - Step 1: Image Processing...")
        os.chdir("Part_I_Image_Splines")
        os.system("python image_processing.py")
        os.chdir("..")
        print("✓ Image processing completed")
    
    print("\\nRunning Part I - Step 2: Cubic Splines...")
    os.chdir("Part_I_Image_Splines")
    os.system("python cubic_splines.py")
    os.chdir("..")
    print("✓ Cubic splines completed")
    
    print("\\nRunning Part II: RLC Circuit Analysis...")
    os.chdir("Part_II_RLC_Circuit")
    os.system("python rlc_analysis.py")
    os.chdir("..")
    print("✓ RLC analysis completed")
    
    print("\\n" + "=" * 60)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Check the generated files:")
    print("- Part_I_Image_Splines/: Ada's spline work")
    print("- Part_II_RLC_Circuit/: Circuit analysis")
    print("- Documentacion/: Spanish documentation")

if __name__ == "__main__":
    main()
