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
import math

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


@dataclass
class StablityOptions:
    use_stability_conditions:bool = False
    """Use stability for declaring existance of edge"""
    stability_period_sec:int = 120
    """ The priod of time to measure an edge stability"""
    stability_rate:float = 0.7
    """ The rate of connectivity in the period of time to declare an edge to be stable"""
    reported_is_stable: bool = False
    """Use this to simulate aggregation of stability in the reports"""

def connected_components(edges):
    G = nx.from_numpy_matrix(edges)
    components = list(nx.connected_components(G))
    return components

class StabilityTracker:
    def __init__(self,steps_window:int,stability_thresh:int):
        self.steps_window=steps_window
        self.stability_thresh=stability_thresh
        self.cmatrices=[]
    
    def report_measurement(self,cmatrix:np.ndarray):
        self.cmatrices.append(cmatrix[...,None])
        if len(self.cmatrices)>self.steps_window:
            self.cmatrices.remove(self.cmatrices[0])
    def get_stable_connectivity(self):
        cmatrix = np.concatenate(self.cmatrices,axis=-1) #TODO: maybe we can skip the concatenate
        cmatrix_count = np.sum(cmatrix,axis=-1)
        return cmatrix_count>=self.stability_thresh

class NetworkSimulator:
    def __init__(self,update_rate:float = 60*3,simulation_rate: float = 1,
                 distribution_params:rnet.NodesDistributionParams  = rnet.NodesDistributionParams(), 
                 frequency:float=200.0  , 
                 propogation_model:PropogationModel =PropogationModelFreeSpace() , 
                 loss_std=7,
                 steps_count = 100,output_video_opts:Optional[RenderingOptions]=None,
                 stability:StablityOptions = StablityOptions()) :
        """Netork main simulator object, simulate the network connections over time and output statistics and video

        Args:
            update_rate (float, optional): The time interval for the models to update their RSSIs & SNRs in the network in secs. Defaults to 60*3.
            simulation_rate (float, optional): The time interval in secs between every step of the simulation. Defaults to 1.
            distribution_params (rnet.NodesDistributionParams, optional): Controls how the nodes are distributed in the defined area. Defaults to rnet.NodesDistributionParams().
            frequency (float, optional): The frequecy in MHz of the transmissions. Defaults to 200.0.
            stability_condition_sec (float): The condition to declare an endge to be stable
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
        self.stability = stability
        
        stability_steps = math.ceil(stability.stability_period_sec/self.simulation_rate)

        self.stability_tracker = StabilityTracker(stability_steps,math.ceil(stability_steps*stability.stability_rate))


        self.current_time =0
        self.callbacks=[]
        self.propogation_model = propogation_model
        self.loss_std = loss_std
        self.set_nodes_turn_on_period()
    
    def set_nodes_turn_on_period(self):
        min_turn_on = 0
        max_turn_on = 3*60
        if self.stability.use_stability_conditions:
            min_turn_on+=self.stability.stability_period_sec
            max_turn_on+=self.stability.stability_period_sec
        for node in self.nodes:
            node.next_update = min_turn_on+np.random.random(1)*(max_turn_on-min_turn_on)
        
        self.max_turn_on = max_turn_on
        

    def full_simulation(self):
        steps_count = self.steps_count
        output_video_opts = self.output_video_opts
        self.real_plus_reported_edges = 0
        self.real_reported_edges = 0
        self.total_reported_edges = 0
        self.total_real_edges = 0

        self.islands_accuracy_nom = 0
        self.islands_accuracy_denom=0

        # Iterate over the range with tqdm
        if output_video_opts is not None:
            fig = plt.figure()
            image_size = (640, 480)  # Adjust the size as per your image dimensions
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For MP4 format, use 'mp4v' or 'avc1'
            video_writer = cv2.VideoWriter(output_video_opts.file_name, fourcc, output_video_opts.fps, image_size)

        with tqdm(range(steps_count), desc="Processing") as pbar:
            for _ in pbar:
                self.step()

                components_real, components_reported = self.calculate_network_component()

                stats = self.calculate_statistics(len(components_real),len(components_reported))

                pbar.set_postfix({"accuracy":f"{stats.accuracy:0.3}","precision":stats.precision,"recall":stats.recall,"islands_accuracy":stats.islands_accuracy})
                if output_video_opts is not None:
                    self.output_netork_to_video(output_video_opts, fig, video_writer, stats,len(components_real))

        if output_video_opts is not None:
            video_writer.release()
        return stats

    def calculate_statistics(self,len_of_components_real,len_of_components_reported):
        if self.current_time<self.max_turn_on:
            # We multiply by to to give the nodes the opportunity to be turned on statistics
            return SimulationStatistics(accuracy=1.0,precision=1.0,recall=1.0,islands_accuracy=1.0)
        self.real_plus_reported_edges+= np.sum((self.current_connectivity | self.reported_connectivity))
        self.real_reported_edges+= np.sum(self.current_connectivity & self.reported_connectivity)
        self.total_reported_edges+= np.sum(self.reported_connectivity)
        self.total_real_edges+=np.sum(self.current_connectivity)
        accuracy = self.real_reported_edges/self.real_plus_reported_edges
        precision = self.real_reported_edges/max(1,self.total_reported_edges)
        recall = self.real_reported_edges / self.total_real_edges

        self.islands_accuracy_nom += min(len_of_components_real,len_of_components_reported)
        self.islands_accuracy_denom += max(len_of_components_real,len_of_components_reported)
        islands_accuracy = self.islands_accuracy_nom/self.islands_accuracy_denom

        stats = SimulationStatistics(accuracy=float(accuracy),precision=float(precision),recall=float(recall),islands_accuracy=float(islands_accuracy))
        return stats

    def calculate_network_component(self):
        current_connectivity = self.current_stable_connectivity & self.current_stable_connectivity.T
        components_real = connected_components(current_connectivity)

        reported_connectivity = self.reported_connectivity & self.reported_connectivity.T
        components_reported = connected_components(reported_connectivity)
        return components_real,components_reported

    def output_netork_to_video(self, output_video_opts, fig, video_writer, stats:SimulationStatistics,num_of_islands):
        fig.clear()
        ax=plt.subplot(111)
        nt.visualize_nodes(self.nodes,fig,is_plotly=False)
        nt.visualize_clusters(self.nodes,fig,is_plotly=False)
        edges_exist=nt.visualize_cmatrix(self.nodes,self.current_connectivity,fig,reported_cmatrix=self.reported_connectivity,is_plotly=False,show_false=output_video_opts.show_false_reports,show_missed=output_video_opts.show_missed_reports)
        time_min = self.current_time//60
        time_sec = self.current_time- time_min*60
        ax.set_title(f"time ={time_min}:{time_sec} , accuracy = {stats.accuracy:0.3} , precision = {stats.precision:0.3} , recall = {stats.recall:0.3} \n islands_accuracy={stats.islands_accuracy:0.3} , number of islands = {num_of_islands}")
        ax.set_ylim(0,self.distribution_params.area_size_y+2*self.distribution_params.margin_y)
        ax.set_xlim(0,self.distribution_params.area_size_x+2*self.distribution_params.margin_x)
        if edges_exist:
            ax.legend()
        canvas = fig.canvas
        canvas.draw()
        image_array = np.array(canvas.renderer.buffer_rgba())
        image_array = image_array[:,:,0:3] # to rgb
        image_array = image_array[:,:,[2,1,0]]
        video_writer.write(image_array)

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

        if self.stability.use_stability_conditions:
            self.stability_tracker.report_measurement(self.current_connectivity)
            self.current_stable_connectivity = self.stability_tracker.get_stable_connectivity()
        else:
            self.current_stable_connectivity = self.current_connectivity

    def update_reported_measurements(self):
        for idx,node in enumerate(self.nodes):
            if node.next_update<=self.current_time:
                self.reported_rssi[idx,:]=self.current_rssi[idx,:] # Maybe add some error here
                self.reported_snr[idx,:]=self.current_snr[idx,:]
                if self.stability.reported_is_stable:
                    self.reported_connectivity[idx,:]=self.current_stable_connectivity[idx,:]
                else:
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

        

