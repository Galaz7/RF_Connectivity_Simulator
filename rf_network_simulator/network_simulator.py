from typing import Optional, Tuple
import numpy as np
from dataclasses import dataclass
from itertools import product
from . import propogation_models as pmodels
from .propogation_models import PropogationModel, PropogationModelFreeSpace
from . import rf_network as rnet
from tqdm import tqdm
from matplotlib import pyplot as plt
from rf_network_simulator import notebook_tools as nt
import cv2

@dataclass
class SimulationStatistics:
    accuracy:float
    """[number of connections that are both real and reported] / [ number of union of real connections & reported connnections] """

@dataclass
class RenderingOptions:
    file_name:str
    fps: float = 8

class NetworkSimulator:
    def __init__(self,update_rate:float = 60*3,simulation_rate: float = 1,distribution_params:rnet.NodesDistributionParams  = rnet.NodesDistributionParams(), frequency:float=200.0  , propogation_model:PropogationModel =PropogationModelFreeSpace()) :
        """Netork main simulator object, simulate the network connections over time and output statistics and video

        Args:
            update_rate (float, optional): The time interval for the models to update their RSSIs & SNRs in the network in secs. Defaults to 60*3.
            simulation_rate (float, optional): The time interval in secs between every step of the simulation. Defaults to 1.
            distribution_params (rnet.NodesDistributionParams, optional): Controls how the nodes are distributed in the defined area. Defaults to rnet.NodesDistributionParams().
            frequency (float, optional): The frequecy in MHz of the transmissions. Defaults to 200.0.
        """
        self.update_rate = update_rate
        self.simulation_rate = simulation_rate
        self.distribution_params = distribution_params
        self.nodes = rnet.create_nodes_samples(distribution_params,frequency)
        l = len(self.nodes)
        self.reported_rssi= -1000*np.ones((l,l))
        self.reported_snr= -1000*np.ones((l,l))
        self.reported_connectivity= np.ones((l,l))==0

        self.current_rssi= -1000*np.ones((l,l))
        self.current_snr= -1000*np.ones((l,l))
        self.current_connectivity= np.ones((l,l))==0

        self.current_time =0
        self.callbacks=[]
        self.propogation_model = propogation_model
    
    def full_simulation(self,steps_count = 100,output_video_opts:Optional[RenderingOptions]=None):
        real_plus_reported_edges = 0
        real_reported_edges = 0


        # Iterate over the range with tqdm
        if output_video_opts is not None:
            fig = plt.figure()
            image_size = (640, 480)  # Adjust the size as per your image dimensions
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For MP4 format, use 'mp4v' or 'avc1'
            video_writer = cv2.VideoWriter(output_video_opts.file_name, fourcc, output_video_opts.fps, image_size)

        with tqdm(range(steps_count), desc="Processing") as pbar:
            for _ in pbar:
                self.step()
                real_plus_reported_edges+= np.sum((self.current_connectivity | self.reported_connectivity))
                real_reported_edges+= np.sum(self.current_connectivity & self.reported_connectivity)
                accuracy = real_reported_edges/real_plus_reported_edges
                pbar.set_postfix({"accuracy":f"{accuracy:0.3}"})
                if output_video_opts is not None:
                    fig.clear()
                    ax=plt.subplot(111)
                    nt.visualize_nodes(self.nodes,fig,'blue',is_plotly=False)
                    nt.visualize_cmatrix(self.nodes,self.current_connectivity,fig,reported_cmatrix=self.reported_connectivity,is_plotly=False)
                    ax.set_title(f"time ={self.current_time}s")
                    ax.set_ylim(0,self.distribution_params.area_size_y)
                    ax.set_xlim(0,self.distribution_params.area_size_x)
                    canvas = fig.canvas
                    canvas.draw()
                    image_array = np.array(canvas.renderer.buffer_rgba())
                    image_array = image_array[:,:,0:3] # to rgb
                    image_array = image_array[:,:,[2,1,0]]
                    video_writer.write(image_array)

        if output_video_opts is not None:
            video_writer.release()
        return SimulationStatistics(accuracy=accuracy)

    def update_current_measurements(self):
        self.current_rssi=rnet.create_recieve_power_matrix(self.nodes,self.propogation_model)

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

        

