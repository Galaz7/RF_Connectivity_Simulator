from enum import Enum
from typing import Optional
import numpy as np
from numpy import ndarray
from abc import ABC,abstractmethod
import pyitm

class ModelTypes(Enum):
    PMODEL_FREE_SPACE = 0
    PMODEL_LONGLY_RICE_SIMPLE = 1

class PropogationModel(ABC):
    @abstractmethod
    def __call__(self,loc1:np.ndarray,loc2:np.ndarray,frequency:float,antenna_height1:Optional[np.ndarray] = None ,antenna_height2:Optional[np.ndarray] = None) -> np.ndarray:
        """Calculates the propogation loss between two points

        Args:
            loc1 (np.ndarray): The x,y location of the first node (in KM)
            loc2 (np.ndarray): The x,y location of the second node (in KM)
            frequency (float): The frequency in MHz
            antenna_height1 (float, optional): The first antenna height above ground (m). Defaults to 0.
            antenna_height2 (float, optional): The second antenna height above ground (m). Defaults to 0.
        """
        pass

def free_space_path_loss(distance_km, frequency_mhz):
    # Convert distance from kilometers to meters
    distance_m = distance_km * 1000

    # Convert frequency from megahertz to hertz
    frequency_hz = frequency_mhz * 1e6

    # Calculate the free space path loss using the Friis transmission equation
    path_loss_db = 20 * np.log10(distance_m) + 20 * np.log10(frequency_hz) + 20 * np.log10(4 * np.pi / 3e8)

    return path_loss_db


class PropogationModelFreeSpace(PropogationModel):
    def __call__(self,loc1:np.ndarray,loc2:np.ndarray,frequency:float,antenna_height1:Optional[np.ndarray] = None ,antenna_height2:Optional[np.ndarray] = None):
        d = np.sqrt(np.sum((loc1-loc2)**2,axis=1))
        return free_space_path_loss(d,frequency)
    


class PropogationModelLonglyRice(PropogationModel):
    def __init__(self,height_map:np.ndarray,map_size:tuple[float,float]):
        """Calculates the RF propagation loss using the Longley-Rice Irregular Terrain Model (ITM) as implemeted in the pyitm package. Without using height map
           See https://pyitm.readthedocs.io/en/latest/api.html for more details on the pacakge

        Args:
            height_map (np.ndarray): height map in meters over the defined area.
            map_size (tuple[float,float]): Map size in KMs (y_size,x_size)
        """
        self.height_map = height_map
        self.map_size = map_size

    def get_segment_index(self,total_size,point_count:int,x:np.ndarray):
        x_norm:np.ndarray = x/total_size
        seg_size = 1.0/(point_count-1)
        x_norm = x_norm/seg_size

        x_idx = x_norm.astype(int)
        x_wgt = x_norm-x_idx
        return x_idx,x_wgt


    def get_terrain_height(self,x:float,y:float):
        x_idx,x_wgt = self.get_segment_index(self.map_size[1],self.height_map.shape[1],x)
        y_idx,y_wgt = self.get_segment_index(self.map_size[0],self.height_map.shape[0],y)

        ul = self.height_map[y_idx+0,x_idx+0]
        ur = self.height_map[y_idx+0,x_idx+1]
        dl = self.height_map[y_idx+1,x_idx+0]
        dr = self.height_map[y_idx+1,x_idx+1]

        result = ul*(1-x_wgt)*(1-y_wgt)+\
                 ur*(x_wgt  )*(1-y_wgt)+\
                 dl*(1-x_wgt)*(  y_wgt)+\
                 dr*(  x_wgt)*(  y_wgt)
        return result

    def __call__(self,loc1:np.ndarray,loc2:np.ndarray,frequency:float,antenna_height1:Optional[np.ndarray] = None ,antenna_height2:Optional[np.ndarray] = None) -> ndarray:
        # Get terrain height at transmitter and receiver locations
        terrain_height1 = self.get_terrain_height(loc1[...,0],loc1[...,1])
        terrain_height2 = self.get_terrain_height(loc2[...,0],loc2[...,1])

        # Setting up the Longley-Rice parameters
        freq_mhz = frequency  # frequency in MHz
        tx_height_m = antenna_height1 + terrain_height1  # transmitter height in meters
        rx_height_m = antenna_height2+terrain_height2  # receiver height in meters
        deltaH = 90  # terrain irregularity parameter [m] #TODO: need to calcualte this
        TSiteCriteria = 1  # Tx Antenna deployment sitting criteria
        RSiteCriteria = 1  # Rx Antenna deployment sitting criteria
        dielectric = 15  # relative permittivity
        conductivity = 0.005  # in S/m
        refractivity = 301  # Surface Refractivity [250 - 400 N-units]
        radio_climate = 5  # Continental Temperate
        polarization = 1  # Vertical polarization
        pctTime = 0.50
        pctLoc = 0.50
        pctConf = 0.50
        ModVar = 1  # Individual mode
        distance_km = float(np.sqrt(np.sum((loc1-loc2)**2)))  # distance between transmitter and receiver in km

        reslen= loc1.shape[0]
        result = np.zeros(reslen)
        # Perform the Longley-Rice calculation
        for i in range(reslen):
            item_res = pyitm.itm.ITMAreadBLoss(
                ModVar,
                deltaH,
                tx_height_m[i],
                rx_height_m[i],
                distance_km[i],
                TSiteCriteria,
                RSiteCriteria,
                dielectric,
                conductivity,
                refractivity,
                freq_mhz,
                radio_climate,
                polarization,
                pctTime,
                pctLoc,
                pctConf
            )
            result[i]=item_res
        return result
        
    
    




