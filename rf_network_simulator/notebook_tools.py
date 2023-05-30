from itertools import product
import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass
import numpy as np
import plotly.graph_objects as go
from . import rf_network as rn

def visualize_nodes_plotly(nodes:list[rn.Node],fig,nodes_color):
    x_list = np.array([node.x for node in nodes])
    y_list = np.array([node.y for node in nodes])
    IDs = [f"index:{i}" for i in range(len(nodes))]
    scatter_trace = go.Scatter(x=x_list, y=y_list, mode='markers',name='Nodes',marker_color=nodes_color,text=IDs)
    fig.add_trace(scatter_trace)

def visualize_nodes_matplot(nodes:list[rn.Node],fig,nodes_color):
    x_list = np.array([node.x for node in nodes])
    y_list = np.array([node.y for node in nodes])
    ax = fig.get_axes()[0]

    ax.plot(x_list, y_list, linestyle='',marker='o',markersize=3,markeredgecolor='black',markeredgewidth=0.7,markerfacecolor=nodes_color)


def visualize_nodes(nodes:list[rn.Node],fig,nodes_color:str,is_plotly:bool=True):
    """Visualizes the nodes in a plotly / matplot lib figure. Only nodes positions

    Args:
        nodes (list[rn.Node]): The list of nodes
        fig : The plotly/matplotlib figure object
        nodes_color (str): The face color of all nodes in the graph
        is_plotly (bool, optional): Set to false if you want to use matplotlib instead of plotly. Defaults to True.
    """
    func = visualize_nodes_plotly if is_plotly else visualize_nodes_matplot
    func(nodes,fig,nodes_color)

def decode_edge_type(real_edge:bool,reported_edge:bool) -> int:
    if real_edge and reported_edge:
        return 0
    elif real_edge:
        return 1
    else: 
        return 2

def enumerate_edges_to_display(nodes:list[rn.Node],real_cmatrix:np.ndarray,edges_color:str|tuple[str,str,str],reported_cmatrix:np.ndarray):
    """Go over the list of nodes, for every node that there is a connection between the two nodes, output the 
       location vector of the two nodes, the edge color classification (real connection, fake connection, missed detection)

    Args:
        nodes (list[rn.Node]): The list of nodes
        real_cmatrix (np.ndarray): The connectivity matrix of the real state
        edges_color (str|tuple[str,str,str]): The color setting for the edges type 
        reported_cmatrix (np.ndarray): The connectivity matrix of the reported state

    Yields:
        x_vec,y_vec,edge_color: The x vector,y vector of the edge, and the edge color
    """
    l = len(nodes)
    if isinstance(edges_color,str):
        edges_color = [edges_color,edges_color,edges_color]

    if reported_cmatrix is None:
        reported_cmatrix= real_cmatrix

    for idx1,idx2 in product(range(l),range(l)):
        if idx1==idx2:
            continue
        real_edge = real_cmatrix[idx1,idx2] and real_cmatrix[idx2,idx1]
        reported_edge = reported_cmatrix[idx1,idx2] and reported_cmatrix[idx2,idx1]
        edge_type = decode_edge_type(real_edge,reported_edge)
        edge_color = edges_color[edge_type]
        if real_edge or reported_edge:
            x1 = nodes[idx1].x
            y1 = nodes[idx1].y
            x2 = nodes[idx2].x
            y2 = nodes[idx2].y
            yield [x1,x2],[y1,y2],edge_color,edge_type

def visualize_cmatrix(nodes:list[rn.Node],real_cmatrix:np.ndarray,fig,edges_color = ('black','red','orange'),reported_cmatrix:np.ndarray=None,is_plotly=True):
    legend_text=['True report','Missed report',"False report"]
    legend_done={}
    for x,y,edge_color,edge_type in enumerate_edges_to_display(nodes,real_cmatrix,edges_color,reported_cmatrix):
        if is_plotly:
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color=edge_color),showlegend=False))
        else:
            ax = fig.get_axes()[0]
            label = None
            if edge_type not in legend_done:
                label=legend_text[edge_type]
                legend_done[edge_type] = True
            ax.plot(x, y, linestyle='-',marker='',color=edge_color,label=label)




def figure_visualize_nodes(nodes,fig=None,nodes_color='blue',edges_color='black',is_plotly=True):
    if fig is None:
        if is_plotly:
            fig = go.Figure()
        else:
            fig = plt.figure()
            plt.subplot(111)

    cmatrix = rn.create_connectivity_matrix(nodes)

    visualize_nodes(nodes,fig,nodes_color,is_plotly)
    visualize_cmatrix(nodes,cmatrix,fig,edges_color,is_plotly=is_plotly)
    # Update the layout
    if is_plotly:
        fig.update_layout(
            title='Nodes',
            xaxis_title='km',
            yaxis_title='km',
            width=1000,
            height=1000
        )
    else:
        fig.legend()
    return fig