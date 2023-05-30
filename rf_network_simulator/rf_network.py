from typing import Tuple
import numpy as np
from dataclasses import dataclass
import numpy as np
from itertools import product
from . import propogation_models as pmodels

@dataclass
class Node:
    id : int
    x : float
    """ x position in the area [km]"""
    y : float
    """ y position in the area [km]"""
    trans_power : float = 30
    """Transmit power in dBm"""
    sensitivity : float = -50
    """Sensitivity in dBm"""
    antenna_gain : float = 1
    """anntenna gain in dbi"""
    antenna_height: float = 3
    """height in [m] from the floor"""
    frequency: float = 100
    """ frequency of transmit/recieve in MHz"""
    velocity: Tuple[float,float] = (0,0)
    """ velocity in [m/s]"""
    next_update: float = 0
    """next update time of the node. used in the simulation to indicate the next time [sec] the node reports its state (rssi/etc)"""
    noise_floor: float = -114 #TODO: Support different reciever types
    """The noise floor of the reciever in dBm/Reciever channel width - for calculation of SNR"""


@dataclass
class NodesDistributionParams:
    area_size_x: float = 30
    """area size in km"""
    area_size_y: float = 30
    """area size in km"""
    nodes_minimal_distance:float =  0.5
    """ nodes minimal distance [km] - required for loss model"""
    nodes_count: int = 100
    """ number of nodes in area"""
    velocity_range:Tuple[float,float] = (0,10.0)
    """minimal,maximal velocity in m/s"""

    #TODO: Decide how to make several types of sensitive nodes


def get_random_position(sample_params:NodesDistributionParams):
    x,y = np.random.random(2)
    x = sample_params.area_size_x*x
    y = sample_params.area_size_y*y
    return x,y

def get_last_minimal_distance(x:np.ndarray,y:np.ndarray):
    dx = x[-1]-x[:-1]
    dy = y[-1]-y[:-1]
    d = dx**2 +dy**2
    dmin  = d.min()
    return np.sqrt(dmin)


def create_nodes_samples(sample_params:NodesDistributionParams,frequency:float = 200.0):
    nodes:list[Node] =[]
    x_vec = np.zeros(sample_params.nodes_count)
    y_vec = np.zeros(sample_params.nodes_count)
    x,y = get_random_position(sample_params)
    x_vec[0]=x 
    y_vec[0]=y
    nodes.append(Node(0,x,y))
    for i in range(1,sample_params.nodes_count):
        min_dist = 0
        while min_dist<sample_params.nodes_minimal_distance:
            x,y =get_random_position(sample_params) 
            x_vec[i]=x
            y_vec[i]=y
            min_dist = get_last_minimal_distance(x_vec[:(i+1)],y_vec[:(i+1)])
        v = sample_params.velocity_range[0] + np.random.random(1)*(sample_params.velocity_range[1]-sample_params.velocity_range[0])
        ang = np.random.random(1)*2*np.pi
        vx = float(v*np.cos(ang))
        vy = float(v*np.sin(ang))
        nodes.append(Node(i,x,y,velocity=(vx,vy),frequency=frequency))
    return nodes


def create_distance_matrix(nodes:list[Node]):
    l = len(nodes)
    matrix = np.zeros((l,l))
    for idx1,idx2 in product(range(l),range(l)):
        matrix[idx1,idx2] = (nodes[idx1].x - nodes[idx2].x)**2 + (nodes[idx1].y - nodes[idx2].y)**2

    matrix = np.sqrt(matrix)
    return matrix


def create_recieve_power_matrix(nodes:list[Node]):
    dists = create_distance_matrix(nodes) + 1000*np.eye(len(nodes))
    loss_matrix = pmodels.free_space_path_loss(dists,nodes[0].frequency)
    power_vec = np.array([node.trans_power for node in nodes])
    antenna_gain_vec  = np.array([node.antenna_gain for node in nodes])

    recieve_power = power_vec[:,None] - loss_matrix  + antenna_gain_vec[:,None] +  antenna_gain_vec[None,:]
    recieve_power = (1-np.eye(len(nodes))) * recieve_power + -1000*np.eye(len(nodes))
    return  recieve_power

def create_connectivity_matrix(nodes:list[Node]):
    recieve_power = create_recieve_power_matrix(nodes)
    sensitivities = np.array([node.sensitivity for node in nodes])

    connectivity_matrix = recieve_power >= sensitivities[None,:]
    return connectivity_matrix


def update_nodes_location(nodes:list[Node],time_interval:float = 1.0):
    """Updates the nodes location after time interval


    Args:
        nodes (list[Node]): The nodes to update the location
        time_interval (float, optional): The time interval in seconds. Defaults to 1.0.
    """
    for node in nodes:
        node.x += time_interval * node.velocity[0]/1000
        node.y += time_interval * node.velocity[1]/1000
    