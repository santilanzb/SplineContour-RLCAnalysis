#!/usr/bin/env python3
"""
Numerical Computing Project - Part I
Cubic Spline Interpolation

This module performs cubic spline interpolation of extracted edge points:
1. Extract upper contour points from edge image
2. Compute natural cubic spline coefficients
3. Evaluate and plot the spline
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load edge-detected image
edge_image = cv2.imread('bordes_panda_canny.jpg', cv2.IMREAD_GRAYSCALE)

if edge_image is None:
    print("Error: Could not load edge image. Ensure 'bordes_panda_canny.jpg' exists.")
    exit()

# 2. Find all contours in the edge image
contours, hierarchy = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# 3. Select the main contour, i.e., the upper contour of the image
if not contours:
    print("No contours found in the image.")
    exit()
main_contour = max(contours, key=cv2.contourArea)
print(f"Main contour found with {len(main_contour)} points.")

# 4. Extract points forming the upper contour
upper_contour_points = {}

for point in main_contour:
    x, y = point[0] # Unpack coordinates (x, y)

    # Maintain the smallest 'y' value (topmost)
    if x in upper_contour_points:
        if y < upper_contour_points[x]:
            upper_contour_points[x] = y
    else:
        upper_contour_points[x] = y

# Convert the dictionary to a sorted list of tuples (x, y)
upper_contour_list = sorted(upper_contour_points.items())
print(f"Extracted {len(upper_contour_list)} points for the upper contour.")

# Visualize extracted points
visualization_image = np.zeros_like(edge_image)
for x, y in upper_contour_list:
    cv2.circle(visualization_image, (x, y), 1, (255), -1)

cv2.imshow('Extracted Upper Contour', visualization_image)
cv2.imwrite('contorno_superior_visualizacion.jpg', visualization_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Prepare data for spline interpolation
x_coords = np.array([p[0] for p in upper_contour_list])
y_coords = np.array([p[1] for p in upper_contour_list])

n = len(x_coords) - 1
if n < 1:
    print("At least 2 points are needed for spline interpolation.")
    exit()

# Compute h_i values
h = np.diff(x_coords)

# Build tridiagonal system (Matrix A and Vector B) for M_i
A = np.zeros((n - 1, n - 1))
B = np.zeros(n - 1)

for i in range(n - 1):
    if i > 0:
        A[i, i-1] = h[i]
    A[i, i] = 2 * (h[i] + h[i+1])
    if i < n - 2:
        A[i, i+1] = h[i+1]
    B[i] = 6 * ((y_coords[i+2] - y_coords[i+1]) / h[i+1] - (y_coords[i+1] - y_coords[i]) / h[i])

# Solve for M_1, ..., M_{n-1}
M_interiors = np.linalg.solve(A, B)
M = np.concatenate(([0], M_interiors, [0]))

print(f"Second derivative values (M): {M}")

# Define the cubic spline evaluation function
def evaluate_spline(x_eval, x_points, y_points, M_values):
    """
    Evaluate cubic spline at a given point x_eval.
    """
    idx = np.searchsorted(x_points, x_eval)

    if idx == 0:
        idx = 1
    elif idx == len(x_points):
        idx = len(x_points) - 1

    i = idx - 1
    h_i = x_points[i+1] - x_points[i]

    A_term = (x_points[i+1] - x_eval) / h_i
    B_term = (x_eval - x_points[i]) / h_i

    s_x = (M_values[i] / 6) * (A_term**3 - A_term) * h_i**2 \
          + (M_values[i+1] / 6) * (B_term**3 - B_term) * h_i**2 \
          + y_points[i] * A_term \
          + y_points[i+1] * B_term

    return s_x

# Generate points for the smooth spline curve
x_spline = np.linspace(x_coords.min(), x_coords.max(), 500)
y_spline = np.array([evaluate_spline(x, x_coords, y_coords, M) for x in x_spline])

# Visualize the results
plt.figure(figsize=(10, 6))
plt.plot(x_coords, y_coords, 'o', label='Original Upper Contour Points')
plt.plot(x_spline, y_spline, '-', label='Natural Cubic Spline')
plt.title('Upper Contour Interpolation with Natural Cubic Spline')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
plt.legend()
plt.gca().invert_yaxis() 
plt.show()
