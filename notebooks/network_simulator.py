from typing import Tuple
import numpy as np
from dataclasses import dataclass
import numpy as np
from itertools import product
import propogation_models as pmodels

import rf_network as rnet




class NetworkSimulator:
    def __init__(self,update_rate:float = 60*3,simulation_rate: float = 1,distribution_params:rnet.NodesDistributionParams  = rnet.NodesDistributionParams() ) -> None:
        self.update_rate = update_rate
        self.simulation_rate = simulation_rate
        self.distribution_params = distribution_params
        self.nodes = rnet.create_nodes_samples(distribution_params)
        l = len(self.nodes)
        self.reported_rssi= -1000*np.ones((l,l))
        self.reported_snr= -1000*np.ones((l,l))
        self.reported_connectivity= 0*np.ones((l,l))

        self.current_rssi= -1000*np.ones((l,l))
        self.current_snr= -1000*np.ones((l,l))
        self.current_connectivity= 0*np.ones((l,l))

        self.current_time =0
        self.callbacks=[]


# def create_connectivity_matrix(nodes:list[Node]):
#     recieve_power = create_recieve_power_matrix(nodes)
#     sensitivities = np.array([node.sensitivity for node in nodes])

#     connectivity_matrix = recieve_power >= sensitivities[None,:]
#     return connectivity_matrix

    def update_current_measurements(self):
        self.current_rssi=rnet.create_recieve_power_matrix(self.nodes)

        sensitivities = np.array([node.sensitivity for node in self.nodes])
        noise_floor = np.array([node.noise_floor for node in self.nodes])

        self.current_snr = self.current_rssi - noise_floor[None,:]
        self.current_connectivity = self.current_rssi >= sensitivities[None,:]

    def update_reported_measurements(self):
        for idx,node in enumerate(self.nodes):
            if node.next_update<=self.current_time:
                self.reported_rssi[idx,:]=self.current_rssi[idx,:] # Maybe add some error here
                self.reported_snr[idx,:]=self.current_snr[idx,:]
                self.reported_connectivity[idx,:]=self.current_connectivity[idx,:]
                node_step = np.random.poisson(self.update_rate)
                node.next_update+=node_step




    def step(self):
        self.current_time+=self.simulation_rate
        rnet.update_nodes_location(self.nodes,self.simulation_rate)
        self.update_current_measurements()
        self.update_reported_measurements()
        for callback in self.callbacks:
            callback(self)

    def register_callback(self,callback):
        self.callbacks.append(callback)

        

