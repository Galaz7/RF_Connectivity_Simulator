import numpy as np


def free_space_path_loss(distance_km, frequency_mhz):
    # Convert distance from kilometers to meters
    distance_m = distance_km * 1000

    # Convert frequency from megahertz to hertz
    frequency_hz = frequency_mhz * 1e6

    # Calculate the free space path loss using the Friis transmission equation
    path_loss_db = 20 * np.log10(distance_m) + 20 * np.log10(frequency_hz) + 20 * np.log10(4 * np.pi / 3e8)

    return path_loss_db

