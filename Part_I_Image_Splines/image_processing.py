#!/usr/bin/env python3
"""
Numerical Computing Project - Part I
Image Processing and Edge Detection

This module handles the initial image processing:
- Grayscale conversion
- Canny edge detection
- Edge image saving for subsequent spline processing
"""

import cv2
import numpy as np
import os

def load_and_process_image(image_path='../image.png', low_threshold=100, high_threshold=200):
    """
    Load image and apply Canny edge detection
    
    Args:
        image_path (str): Path to input image
        low_threshold (int): Lower threshold for Canny
        high_threshold (int): Upper threshold for Canny
        
    Returns:
        tuple: (original_image, edge_image)
    """
    print("=" * 60)
    print("PART I: IMAGE PROCESSING AND EDGE DETECTION")
    print("=" * 60)
    print(f"Loading image from: {image_path}")
    
    # Load image in grayscale
    imagen = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if imagen is None:
        print(f"Error: Could not load image. Check path: {image_path}")
        return None, None
    
    print(f"✓ Image loaded successfully")
    print(f"  Dimensions: {imagen.shape[1]} x {imagen.shape[0]} pixels")
    print(f"  Data type: {imagen.dtype}")
    
    # Apply Canny edge detector
    print(f"\nApplying Canny edge detection...")
    print(f"  Lower threshold: {low_threshold}")
    print(f"  Upper threshold: {high_threshold}")
    
    bordes = cv2.Canny(imagen, low_threshold, high_threshold)
    
    print(f"✓ Edge detection completed")
    print(f"  Edge pixels detected: {np.count_nonzero(bordes)}")
    
    return imagen, bordes

def display_results(original_image, edge_image):
    """
    Display original and edge-detected images
    """
    if original_image is None or edge_image is None:
        return
        
    print(f"\nDisplaying results...")
    
    # Show original image in grayscale
    cv2.imshow('Original Image (Grayscale)', original_image)
    
    # Show detected edges
    cv2.imshow('Detected Edges (Canny)', edge_image)
    
    # Wait for key press to exit
    print("Press any key to close windows...")
    cv2.waitKey(0)
    
    # Close all OpenCV windows
    cv2.destroyAllWindows()

def save_edge_image(edge_image, output_path='bordes_panda_canny.jpg'):
    """
    Save the edge-detected image
    """
    if edge_image is None:
        return False
        
    cv2.imwrite(output_path, edge_image)
    print(f"✓ Edge image saved as: {output_path}")
    return True

def main():
    """
    Main function for image processing workflow
    """
    try:
        # Load and process image
        original_image, edge_image = load_and_process_image()
        
        if original_image is None:
            return False
            
        # Display results
        display_results(original_image, edge_image)
        
        # Save edge image
        save_edge_image(edge_image)
        
        print(f"\n✓ Image processing completed successfully!")
        print(f"Next step: Run cubic_splines.py for spline interpolation")
        
        return True
        
    except Exception as e:
        print(f"Error during image processing: {str(e)}")
        return False

if __name__ == "__main__":
    main()
