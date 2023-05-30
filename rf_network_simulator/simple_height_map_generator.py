import scipy.ndimage as ndimage
import numpy as np
from matplotlib import pyplot as plt

def generate_terrain_height_map(size, max_elevation, smoothness):
    """
    Generates a randomized terrain height map for a rural area.

    Args:
        size (tuple): Size of the height map (rows, columns).
        max_elevation (float): Maximum elevation value for the terrain.
        smoothness (float): Controls the smoothness of the terrain. Higher values result in smoother terrain.

    Returns:
        numpy.ndarray: 2D array representing the terrain height map.
    """
    # Generate a random noise map
    noise_map = np.random.uniform(low=0.0, high=1.0, size=size)

    # Smooth the noise map using a Gaussian filter
    smoothed_map = ndimage.gaussian_filter(noise_map, sigma=smoothness)

    # Normalize the smoothed map to the desired elevation range
    normalized_map = max_elevation * (smoothed_map - np.min(smoothed_map)) / (np.max(smoothed_map) - np.min(smoothed_map))

    return normalized_map


def visualize_terrain_height_map(terrain_height_map):
    """
    Visualizes a terrain height map.

    Args:
        terrain_height_map (numpy.ndarray): 2D array representing the terrain height map.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(terrain_height_map, cmap='terrain', origin='lower')
    plt.colorbar(label='Elevation (m)')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.title('Terrain Height Map')
    plt.show()