from typing import Optional, Tuple
import numpy as np
from dataclasses import dataclass
from itertools import product
from . import propogation_models as pmodels
from .propogation_models import PropogationModel, PropogationModelFreeSpace
from . import rf_network as rnet
from tqdm import tqdm
from matplotlib import pyplot as plt
from rf_network_simulator import visualization_tools as nt
import cv2
from eparams import params
import networkx as nx

@params
class SimulationStatistics:
    accuracy:float
    """[number of connections that are both real and reported] / [ number of union of real connections & reported connnections] """
    precision:float
    recall:float
    islands_accuracy:float

@dataclass
class RenderingOptions:
    file_name:str
    fps: float = 8
    show_missed_reports: bool = True
    show_false_reports:bool = True


def connected_components(edges):
    G = nx.from_numpy_matrix(edges)
    components = list(nx.connected_components(G))
    return components

class NetworkSimulator:
    def __init__(self,update_rate:float = 60*3,simulation_rate: float = 1,
                 distribution_params:rnet.NodesDistributionParams  = rnet.NodesDistributionParams(), 
                 frequency:float=200.0  , 
                 propogation_model:PropogationModel =PropogationModelFreeSpace() , 
                 loss_std=7,
                 steps_count = 100,output_video_opts:Optional[RenderingOptions]=None) :
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
        self.steps_count = steps_count
        self.output_video_opts=output_video_opts
        self.current_rssi= -1000*np.ones((l,l))
        self.current_snr= -1000*np.ones((l,l))
        self.current_connectivity= np.ones((l,l))==0

        self.current_time =0
        self.callbacks=[]
        self.propogation_model = propogation_model
        self.loss_std = loss_std
    
    def full_simulation(self):
        steps_count = self.steps_count
        output_video_opts = self.output_video_opts
        real_plus_reported_edges = 0
        real_reported_edges = 0
        total_reported_edges = 0
        total_real_edges = 0

        islands_accuracy_nom = 0
        islands_accuracy_denom=0

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
                total_reported_edges+= np.sum(self.reported_connectivity)
                total_real_edges+=np.sum(self.current_connectivity)
                accuracy = real_reported_edges/real_plus_reported_edges
                precision = real_reported_edges/total_reported_edges
                recall = real_reported_edges / total_real_edges

                current_connectivity = self.current_connectivity & self.current_connectivity.T
                components_real = connected_components(current_connectivity)

                reported_connectivity = self.reported_connectivity & self.reported_connectivity.T
                components_reported = connected_components(reported_connectivity)

                islands_accuracy_nom += min(len(components_real),len(components_reported))
                islands_accuracy_denom += max(len(components_real),len(components_reported))
                islands_accuracy = islands_accuracy_nom/islands_accuracy_denom

                pbar.set_postfix({"accuracy":f"{accuracy:0.3}","precision":precision,"recall":recall,"islands_accuracy":islands_accuracy})
                if output_video_opts is not None:
                    fig.clear()
                    ax=plt.subplot(111)
                    nt.visualize_nodes(self.nodes,fig,is_plotly=False)
                    nt.visualize_clusters(self.nodes,fig,is_plotly=False)
                    nt.visualize_cmatrix(self.nodes,self.current_connectivity,fig,reported_cmatrix=self.reported_connectivity,is_plotly=False,show_false=output_video_opts.show_false_reports,show_missed=output_video_opts.show_missed_reports)
                    time_min = self.current_time//60
                    time_sec = self.current_time- time_min*60
                    ax.set_title(f"time ={time_min}:{time_sec} , accuracy = {accuracy:0.3} , precision = {precision:0.3} , recall = {recall:0.3} \n islands_accuracy={islands_accuracy:0.3} , number of islands = {len(components_real)}")
                    ax.set_ylim(0,self.distribution_params.area_size_y+2*self.distribution_params.margin_y)
                    ax.set_xlim(0,self.distribution_params.area_size_x+2*self.distribution_params.margin_x)
                    ax.legend()
                    canvas = fig.canvas
                    canvas.draw()
                    image_array = np.array(canvas.renderer.buffer_rgba())
                    image_array = image_array[:,:,0:3] # to rgb
                    image_array = image_array[:,:,[2,1,0]]
                    video_writer.write(image_array)

        if output_video_opts is not None:
            video_writer.release()
        return SimulationStatistics(accuracy=float(accuracy),precision=float(precision),recall=float(recall),islands_accuracy=float(islands_accuracy))

    def update_current_measurements(self):
        self.current_rssi=rnet.create_recieve_power_matrix(self.nodes,self.propogation_model)
        
        if self.loss_std>0:
            loss_random=np.random.normal(loc=0,scale=self.loss_std,size=self.current_rssi.shape)
        else:
            loss_random=0
        self.current_rssi += loss_random

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

        

