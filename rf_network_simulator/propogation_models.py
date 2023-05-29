import numpy as np


def free_space_path_loss(distance_km, frequency_mhz):
    # Convert distance from kilometers to meters
    distance_m = distance_km * 1000

    # Convert frequency from megahertz to hertz
    frequency_hz = frequency_mhz * 1e6

    # Calculate the free space path loss using the Friis transmission equation
    path_loss_db = 20 * np.log10(distance_m) + 20 * np.log10(frequency_hz) + 20 * np.log10(4 * np.pi / 3e8)

    return path_loss_db


def itm_propagation_loss(terrain_height_map, tx_location,tx_terrain_height, tx_height, rx_location,rx_terrain_height, rx_height, frequency_mhz):
    """
    Calculates the RF propagation loss using the Longley-Rice Irregular Terrain Model (ITM).

    Args:
        terrain_height_map (numpy.ndarray): 2D array representing the terrain height map.
        tx_location (tuple): Tuple containing the (x, y) coordinates of the transmitter location.
        rx_location (tuple): Tuple containing the (x, y) coordinates of the receiver location.
        tx_terrain_height (float): Terrain height of the transmitter antenna above ground [m].
        tx_height (float): Height of the transmitter antenna in meters.
        rx_terrain_height (tuple): Terrain height of the receiver antenna above ground [m].
        rx_height (float): Height of the receiver antenna in meters.
        frequency_mhz (float): Frequency of the signal in MHz.

    Returns:
        float: Propagation loss in decibels (dB).
    """

    # Get terrain height at transmitter and receiver locations
    tx_terrain_height = terrain_height_map[tx_location[1], tx_location[0]]
    rx_terrain_height = terrain_height_map[rx_location[1], rx_location[0]]

    # Calculate the path distance
    distance = np.sqrt((rx_location[0] - tx_location[0])**2 + (rx_location[1] - tx_location[1])**2)

    # Calculate the elevation angle
    elevation_angle = np.arctan((rx_terrain_height + rx_height - tx_terrain_height - tx_height) / distance)

    # Calculate the path loss using the ITM
    path_loss = 69.55 + 26.16 * np.log10(frequency_mhz) - 13.82 * np.log10(tx_height) \
                - 4.97 * np.log10(tx_height) * np.log10(distance) + (1.1 * np.log10(frequency_mhz) - 0.7) * elevation_angle \
                - 1.56 * np.log10(frequency_mhz) + 0.8

    return path_loss
