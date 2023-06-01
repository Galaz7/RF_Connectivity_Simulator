from eparams import params


import numpy as np

import numpy as np

from rf_network_simulator.network_simulator import NetworkSimulator,RenderingOptions, StablityOptions
from rf_network_simulator.rf_network import NodesDistributionParams
from rf_network_simulator import propogation_models as pmodels
from rf_network_simulator import simple_height_map_generator as hmgen
from rf_network_simulator.rf_network import NodeTypeDistribution
import os
import git
from os import path as osp
import datetime



@params
class SizeParams:
    """The area size to place the nodes on"""
    size_x: float =5.0
    """in km"""
    size_y: float = 5.0
    """in km"""
    area_margin: float = 0.5
    """ Area margins in percentage"""

@params
class VisualizationParams:
    show_missed_reports: bool = False

@params
class StabilityParams:
    use_stability_conditions:bool = False
    """Use stability for declaring existance of edge"""
    stability_period_sec:int = 120
    """ The priod of time to measure an edge stability"""
    stability_rate:float = 0.7
    """ The rate of connectivity in the period of time to declare an edge to be stable"""
    reported_is_stable: bool = False
    """Use this to simulate aggregation of stability in the reports"""

@params
class Config:
    experiment_name:str=""
    seed:int = 0
    propogation_model:str = "egli"
    simulation_rate_secs:float = 2
    """seconds"""
    update_rate_mins:float = 15
    """minutes"""
    simulation_steps_count:int = 240
    area_size:SizeParams = SizeParams()
    number_of_clusters:int =4
    percentage_of_vehicles: float = 0.0
    number_of_total_nodes: int = 40
    person_maximal_velocity: float = 5
    """kph"""
    vehicle_velocity_range: list[float] = [10,20]
    """kph"""
    visualizations: VisualizationParams = VisualizationParams()
    create_video: bool = True
    stability: StabilityParams = StabilityParams()





def create_simulator_from_yaml(params_file_name="params.yaml") ->tuple[NetworkSimulator,str]:

    params = Config()

    if os.path.exists(params_file_name):
        params._from_yaml(params_file_name)


    git_repo = git.Repo(".", search_parent_directories=True)
    project_root = git_repo.git.rev_parse("--show-toplevel")

    experiments_folder=osp.join(project_root,"experiments")

    if not osp.exists(experiments_folder):
        os.makedirs(experiments_folder)
        with open(osp.join(experiments_folder,".gitignore"),'wt') as f:
                f.write("*")


    # Get the current date and time
    current_datetime = datetime.datetime.now()

    # Format the date and time
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M")

    experiment_name = params.experiment_name.replace(" ","_")

    sha = git_repo.head.commit.hexsha[0:8]

    experiment_folder = formatted_datetime + "_" + sha + "__" + experiment_name

    experiment_folder = osp.join(experiments_folder,experiment_folder)
    os.makedirs(experiment_folder)

    params._to_yaml(osp.join(experiment_folder,"params.yaml"))

    if params.percentage_of_vehicles==0:
        node_types=[NodeTypeDistribution(params.number_of_total_nodes,2.5,(0,params.person_maximal_velocity/3.6))]
    else:
        number_of_vehicles = max(1,int(params.number_of_total_nodes * params.percentage_of_vehicles))
        node_types=[NodeTypeDistribution(params.number_of_total_nodes-number_of_vehicles,2.5,(0,params.person_maximal_velocity/3.6)),
                    NodeTypeDistribution(number_of_vehicles,3.5,(params.vehicle_velocity_range[0]/3.6,params.vehicle_velocity_range[1]/3.6))]

    margin_x = params.area_size.size_x*params.area_size.area_margin
    margin_y = params.area_size.size_y*params.area_size.area_margin
    dist = NodesDistributionParams(area_size_x=params.area_size.size_x,area_size_y=params.area_size.size_y,margin_x=margin_x,margin_y=margin_y,nodes_minimal_distance=0.002,node_types=node_types)

    terrain_map = hmgen.generate_terrain_height_map(size=(100, 100), max_elevation=160, smoothness=5) #TODO: consider make this optional

    propogation_models = {
        "free_space": pmodels.PropogationModelFreeSpace(),
        "egli":pmodels.PropogationModelEgli(),
        "longly_rice": pmodels.PropogationModelLonglyRice(terrain_map,(params.area_size.size_x+margin_x,params.area_size.size_y+margin_y))
    }

    propogation_model = propogation_models[params.propogation_model]

    total_time = params.simulation_steps_count #iterations
    update_rate = params.update_rate_mins*60


    if params.seed>=0:
        np.random.seed(params.seed) 


    if params.create_video:
        output_video_opts=RenderingOptions(file_name=osp.join(experiment_folder,"output.mp4"),show_missed_reports=params.visualizations.show_missed_reports)
    else:
        output_video_opts=None

    stability_opts = StablityOptions(params.stability.use_stability_conditions,params.stability.stability_period_sec,params.stability.stability_rate,params.stability.reported_is_stable)

    sim = NetworkSimulator(simulation_rate=params.simulation_rate_secs,update_rate=update_rate,frequency=200,propogation_model=propogation_model,distribution_params=dist,steps_count=total_time,output_video_opts=output_video_opts,stability=stability_opts)
    return sim,experiment_folder

