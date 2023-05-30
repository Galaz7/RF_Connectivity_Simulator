from typing import Tuple
import numpy as np
from dataclasses import dataclass,field
import numpy as np
from itertools import product
from .propogation_models import PropogationModel,PropogationModelFreeSpace

@dataclass
class Node:
    id : int
    type_id : int
    x : float
    """ x position in the area [km]"""
    y : float
    """ y position in the area [km]"""
    trans_power : float = 30
    """Transmit power in dBm"""
    sensitivity : float = -98
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
    noise_floor: float = -114 #TODO: Think on what to do with this and the SNR measurements
    """The noise floor of the reciever in dBm/Reciever channel width - for calculation of SNR"""

# 20 vehicle , 35 manned

# 3.5m (32dBm), 2m (32dBm) 

@dataclass
class NodeTypeDistribution:
    count: int
    """ number of nodes of this type in area"""
    antenna_height:float
    """ antenna height for this type of nodes"""
    velocity_range:Tuple[float,float]
    """minimal,maximal velocity in m/s"""
    trans_power:float = 32
    """ node transmitter power"""
    antenna_gain:float = 1
    """ node antenna gain"""
    node_sensitivity: float = -98
    """node sensitivity"""

@dataclass
class NodesDistributionParams:
    area_size_x: float = 5
    """area size in km"""
    area_size_y: float = 5
    """area size in km"""
    start_x: float =0
    """position offset in km for nodes placement"""
    start_y: float =0
    """position offset in km for nodes placement"""
    nodes_minimal_distance:float =  0.5
    """ nodes minimal distance [km] - required for loss model"""

    # node_types:list[NodeTypeDistribution] = [NodeTypeDistribution(60,2,(0,12.0/3.6))]
    node_types:list[NodeTypeDistribution] = field(default_factory=lambda: [NodeTypeDistribution(20,3.5,(10,30.0/3.6)),NodeTypeDistribution(35,2,(0,5.0/3.6))])



def get_random_position(sample_params:NodesDistributionParams):
    x,y = np.random.random(2)
    x = sample_params.area_size_x*x + sample_params.start_x
    y = sample_params.area_size_y*y + sample_params.start_y
    return x,y

def get_last_minimal_distance(x:np.ndarray,y:np.ndarray):
    dx = x[-1]-x[:-1]
    dy = y[-1]-y[:-1]
    d = dx**2 +dy**2
    dmin  = d.min()
    return np.sqrt(dmin)


def create_nodes_samples(sample_params:NodesDistributionParams,frequency:float = 200.0) -> list[Node]:
    nodes:list[Node] =[]
    nodes_count_list = [node_type.count for node_type in sample_params.node_types]
    nodes_count = sum(nodes_count_list)
    x_vec = np.zeros(nodes_count)
    y_vec = np.zeros(nodes_count)
    x,y =get_random_position(sample_params) 
    id = 0
    for type_id,node_type in enumerate(sample_params.node_types):
        for _ in range(node_type.count):
            min_dist = 0
            while min_dist<sample_params.nodes_minimal_distance and len(nodes)>0:
                x,y =get_random_position(sample_params) 
                x_vec[id]=x
                y_vec[id]=y
                min_dist = get_last_minimal_distance(x_vec[:(id+1)],y_vec[:(id+1)])
            v = node_type.velocity_range[0] + np.random.random(1)*(node_type.velocity_range[1]-node_type.velocity_range[0])
            ang = np.random.random(1)*2*np.pi
            vx = float(v*np.cos(ang))
            vy = float(v*np.sin(ang))
            nodes.append(
                Node(id,type_id,x,y,velocity=(vx,vy),
                     frequency=frequency,sensitivity=node_type.node_sensitivity,
                     antenna_gain=node_type.antenna_gain,antenna_height=node_type.antenna_height,
                     trans_power=node_type.trans_power))
            id+=1
    return nodes


def create_distance_matrix(nodes:list[Node]):
    l = len(nodes)
    matrix = np.zeros((l,l))
    for idx1,idx2 in product(range(l),range(l)):
        matrix[idx1,idx2] = (nodes[idx1].x - nodes[idx2].x)**2 + (nodes[idx1].y - nodes[idx2].y)**2

    matrix = np.sqrt(matrix)
    return matrix


def create_recieve_power_matrix(nodes:list[Node],propogation_model:PropogationModel=PropogationModelFreeSpace()) :
    #dists = create_distance_matrix(nodes) + 1000*np.eye(len(nodes))
    
    idx = np.arange(len(nodes))
    x_idx,y_idx = np.meshgrid(idx,idx)
    x_idx,y_idx = [x.ravel() for x in [x_idx,y_idx]]

    locs =np.array([[node.x,node.y] for node in nodes])
    heights =np.array([node.antenna_height for node in nodes])

    loc1 = locs[x_idx,:]
    loc2 = locs[y_idx,:]
    h1 = heights[x_idx]
    h2 = heights[y_idx]



    loss_matrix = propogation_model(loc1,loc2,nodes[0].frequency,h1,h2)
    loss_matrix = loss_matrix.reshape((len(nodes),len(nodes)))

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
    