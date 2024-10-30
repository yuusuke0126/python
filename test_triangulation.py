import numpy as np
from scipy.optimize import least_squares
from math import sin, cos
import time

# Coordinates of the known points (example)
known_points = np.array([
    [1.0, 1.0],
    [4.0, 1.0],
    [1.0, 4.0],
    [4.0, 4.0]
])

# Detected tags relative to the robot's frame (example)
detected_tag = np.array([
    [2.0, 3.0],
    [3.0, 4.0],
    [0.0, 5.0],
    [-1.0, -1.0]
])

# Define the function to minimize
def residuals(params, points, detected_tags):
    x, y, theta = params
    residuals = []
    for i in range(len(points)):
        xi_ref, yi_ref = points[i]
        xi_tag, yi_tag = detected_tags[i]
        # Transform the detected tags to the global frame considering the robot's position and orientation
        xi = x + xi_tag * cos(theta) - yi_tag * sin(theta)
        yi = y + xi_tag * sin(theta) + yi_tag * cos(theta)
        # Calculate the squared differences
        residuals.append((xi_ref - xi)**2 + (yi_ref - yi)**2)
    return residuals

# Initial guess for the position and orientation
initial_guess = np.array([0.0, 0.0, 0.0])

# Measure the time taken for the least-squares optimization
start_time = time.time()
result = least_squares(residuals, initial_guess, args=(known_points, detected_tag))
end_time = time.time()

# Extract the estimated position and orientation
estimated_position = result.x

# Display the estimated position and orientation
print(f"Estimated Position: (x, y) = ({estimated_position[0]:.2f}, {estimated_position[1]:.2f})")
print(f"Estimated Orientation: theta = {estimated_position[2]:.2f} radians")

# Display the processing time
print(f"Processing Time: {end_time - start_time:.6f} seconds")
