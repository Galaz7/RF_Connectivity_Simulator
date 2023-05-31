import numpy as np
from abc import ABC,abstractmethod

def get_random_position(area_size,start_offset):
    x,y = np.random.random(2)
    x = area_size[0]*x + start_offset[0]
    y = area_size[1]*y + start_offset[1]
    return x,y


def get_last_minimal_distance(x:np.ndarray,y:np.ndarray):
    dx = x[-1]-x[:-1]
    dy = y[-1]-y[:-1]
    d = dx**2 +dy**2
    dmin  = d.min()
    return np.sqrt(dmin)

class SpatialDistributer(ABC):
    @abstractmethod
    def __call__(self)->tuple[float,float,int]:
        pass


class UniformSpatialDistributer(SpatialDistributer):
    def __init__(self,nodes_count,area_size:tuple[int,int],start_offset:tuple[int,int],nodes_minimal_distance) -> None:
        self.x_vec = np.zeros(nodes_count)
        self.y_vec = np.zeros(nodes_count)
        self.nodes_count = nodes_count
        self.area_size=area_size
        self.start_offset=start_offset
        self.nodes_minimal_distance = nodes_minimal_distance
        self.counter = 0
    
    def __call__(self):

        min_dist = 0
        while True:
            x,y =get_random_position(self.area_size,self.start_offset) 
            self.x_vec[self.counter]=x
            self.y_vec[self.counter]=y
            if self.counter>0:
                min_dist = get_last_minimal_distance(self.x_vec[:(self.counter+1)],self.y_vec[:(self.counter+1)])
            if min_dist>self.nodes_minimal_distance or self.counter==0:
                break
        self.counter+=1
        return x,y,(self.counter-1)


class ClusteredDistributer(SpatialDistributer):
    def __init__(self,nodes_count,area_size:tuple[int,int],start_offset:tuple[int,int],nodes_minimal_distance:float,number_of_clusters:int=4,clusters_minimal_distance:float=1.0,cluster_radius:float=1.0) -> None:
        self.x_vec = np.zeros(nodes_count)
        self.y_vec = np.zeros(nodes_count)
        self.nodes_count = nodes_count
        self.area_size=area_size
        self.start_offset=start_offset
        self.nodes_minimal_distance = nodes_minimal_distance
        self.counter = 0
        self.number_of_clusters = number_of_clusters
        self.clusters_minimal_disance=clusters_minimal_distance
        self.cluster_centers = self.randomize_clusters_centers()
        self.current_cluster_index = 0
        self.cluster_radius = cluster_radius
    
    def randomize_clusters_centers(self):
        #TODO: think of somthing more suphisticated . Myabe ignore the 5km limit for this and add nodes one at a time - to not be stuck on a loop
        dist = UniformSpatialDistributer(nodes_count=self.number_of_clusters, area_size=self.area_size, start_offset=self.start_offset,nodes_minimal_distance=self.clusters_minimal_disance)
        cluster_centers =[]
        for _ in range (self.number_of_clusters):
            x,y,_ = dist()
            cluster_centers.append([x,y])
        return cluster_centers

    def get_current_cluster_index(self):
        index = self.current_cluster_index
        self.current_cluster_index+=1
        if self.current_cluster_index>= self.number_of_clusters:
            self.current_cluster_index=0
        return index
            


    def __call__(self):
        x,y =get_random_position(self.area_size,self.start_offset) 

        min_dist = 0
        cluster_size =[2*self.cluster_radius,2*self.cluster_radius]
        cluster_index = self.get_current_cluster_index()
        while True:
            c_x,c_y = self.cluster_centers[cluster_index]
            cluster_offset = [c_x-self.cluster_radius,c_y-self.cluster_radius]
            x,y =get_random_position(cluster_size,cluster_offset) 
            self.x_vec[self.counter]=x
            self.y_vec[self.counter]=y
            if self.counter>0:
                min_dist = get_last_minimal_distance(self.x_vec[:(self.counter+1)],self.y_vec[:(self.counter+1)])
            if min_dist>=self.nodes_minimal_distance or self.counter==0:
                break

        self.counter+=1
        return x,y,cluster_index

class VelocityDistributer:
    def __init__(self,uncertainty = 5):
        """Distribute a random velocity with same angle for cluster

        Args:
            uncertainty (int, optional): uncertainty range in degrees for nodes in cluster (the angle of each element will be randomized with this uncertainty). Defaults to 10.
        """
        self.cluster_velocity_directions={}
        self.uncertainty = uncertainty

    def __call__(self,cluster_index,velocity_range):
        
        if cluster_index in self.cluster_velocity_directions:
            angle = self.cluster_velocity_directions[cluster_index]
        else:
            angle = 2*np.pi*(22.5+np.random.random(1)*45)/360
            self.cluster_velocity_directions[cluster_index] = angle

        angle_uncertainty = 2*np.pi*(np.random.random(1)*2*self.uncertainty - self.uncertainty)/360
        dir = np.array([np.sin(angle+angle_uncertainty),np.cos(angle+angle_uncertainty)])
        v = velocity_range[0] + np.random.random(1)*(velocity_range[1]-velocity_range[0])
        velocity = v*dir
        return velocity