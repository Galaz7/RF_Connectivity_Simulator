from abc import ABC,abstractmethod
from enum import Enum

import numpy as np

from rf_network_simulator.rf_network import Node

class VelocityModelType(Enum):
    VMODEL_SIMPLE = 0
    VMODEL_CLUSTER = 1

class VelocityModel(ABC):
    
    @abstractmethod
    def update_nodes_locations(self,nodes: list[Node],time_interval:float):
        pass

class VelocityModelSimple(VelocityModel):
    def update_nodes_locations(self,nodes: list[Node],time_interval:float):
        for node in nodes:
            node.x += time_interval * node.velocity[0]/1000
            node.y += time_interval * node.velocity[1]/1000        


class VelocityModelClusters(VelocityModel):
    def __init__(self,nodes: list[Node],max_inner_cluster_radius:float,max_outer_radius) -> None:
        super().__init__()
        clusters = set([node.cluster_index for node in nodes])
        self.cluster_ids = sorted(clusters)
        cluster_velocity_list=[]
        for cluster_id in clusters:
            velocity=np.array([node.velocity for node in nodes if node.cluster_index==cluster_id])
            velocity = np.mean(velocity)
            cluster_velocity_list.append(velocity)
        self.cluster_velocity = np.array(cluster_velocity_list)
        self.total_velocity = np.mean(self.cluster_velocity,axis=0)
        self.max_inner_cluster_radius = max_inner_cluster_radius
        self.max_outer_radius = max_outer_radius
        
    def get_clusters_centers(self,nodes:list[Node]):
        clusters_centers = []
        for cluster_id in self.cluster_ids:
            locations = np.array([[node.x,node.y] for node in nodes if node.cluster_index == cluster_id])
            center=np.mean(locations,axis=0)
            clusters_centers.append(center)
        return clusters_centers


    def update_nodes_locations(self,nodes: list[Node],time_interval:float):
        clusters_centers = self.get_clusters_centers(nodes)
        amp_spread_min = 0.05
        amp_spread_max = 0.15
        deg_spread=5*np.pi/180
        for node in nodes:
            ccenter = clusters_centers[node.cluster_index]
            loc = np.array([node.x,node.y])
            d= np.sqrt(np.sum((loc-ccenter)**2))
            if d>self.max_inner_cluster_radius:
                to_center = ccenter-loc
                angle = np.arctan2(to_center[0],to_center[1])
                angle = angle -deg_spread + 2*deg_spread*np.random.random(1)

                cvelocity = self.cluster_velocity[node.cluster_index]
                amp = (amp_spread_min+(amp_spread_max-amp_spread_min)*np.random.random(1))*np.linalg.norm(cvelocity)
                vx = np.cos(angle)
                vy = np.sin(angle)
                angle_unity = np.array([vx,vy])

                v_new = cvelocity + angle_unity*amp
                node.velocity=(float(v_new[0]),float(v_new[1]))

                
            node.x += float(time_interval * node.velocity[0]/1000)
            node.y += float(time_interval * node.velocity[1]/1000)
