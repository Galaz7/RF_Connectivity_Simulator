from typing import Tuple
import numpy as np
from dataclasses import dataclass,field
import numpy as np
from itertools import product
from .propogation_models import PropogationModel,PropogationModelFreeSpace
from .spatial_distribution_model import ClusteredDistributer, UniformSpatialDistributer, VelocityDistributer

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
    frequency: float = 200
    """ frequency of transmit/recieve in MHz"""
    velocity: Tuple[float,float] = (0,0)
    """ velocity in [m/s]"""
    next_update: float = 0
    """next update time of the node. used in the simulation to indicate the next time [sec] the node reports its state (rssi/etc)"""
    noise_floor: float = -114 #TODO: Think on what to do with this and the SNR measurements
    """The noise floor of the reciever in dBm/Reciever channel width - for calculation of SNR"""

    cluster_index: int = 0

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
    margin_x: float =0
    """position offset in km for nodes placement"""
    margin_y: float =0
    """position offset in km for nodes placement"""
    nodes_minimal_distance:float =  0.5
    """ nodes minimal distance [km] - required for loss model"""

    # node_types:list[NodeTypeDistribution] = [NodeTypeDistribution(60,2,(0,12.0/3.6))]
    node_types:list[NodeTypeDistribution] = field(default_factory=lambda: [NodeTypeDistribution(20,3.5,(10,30.0/3.6)),NodeTypeDistribution(35,2,(0,5.0/3.6))])


def create_nodes_samples(sample_params:NodesDistributionParams,frequency:float = 200.0) -> list[Node]:
    nodes:list[Node] =[]
    nodes_count_list = [node_type.count for node_type in sample_params.node_types]
    nodes_count = sum(nodes_count_list)
    id = 0
    area_size=(sample_params.area_size_x,sample_params.area_size_y)
    start_offset = (sample_params.margin_x,sample_params.margin_y)
    spat_dist = ClusteredDistributer(nodes_count=nodes_count,area_size=area_size, start_offset=start_offset,nodes_minimal_distance=sample_params.nodes_minimal_distance)
    velocity_dist = VelocityDistributer()
    for type_id,node_type in enumerate(sample_params.node_types):
        for _ in range(node_type.count):
            x,y,cluster_index = spat_dist()
            v = velocity_dist(cluster_index,node_type.velocity_range)            
            nodes.append(
                Node(id,type_id,x,y,velocity=(v[0],v[1]),
                     frequency=frequency,sensitivity=node_type.node_sensitivity,
                     antenna_gain=node_type.antenna_gain,antenna_height=node_type.antenna_height,
                     trans_power=node_type.trans_power,cluster_index=cluster_index))
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

    locs =np.array([[node.x,node.y] for node in nodes]).squeeze() #TODO: Need to check why there is an extra dimension here...
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
    