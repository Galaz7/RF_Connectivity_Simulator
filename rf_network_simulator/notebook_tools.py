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


def visualize_nodes(nodes:list[rn.Node],fig,nodes_color,is_plotly=True):
    func = visualize_nodes_plotly if is_plotly else visualize_nodes_matplot
    func(nodes,fig,nodes_color)

def enumerate_edges_to_display(nodes:list[rn.Node],real_cmatrix:np.ndarray,fig,edges_color,reported_cmatrix:np.ndarray):
    l = len(nodes)
    if isinstance(edges_color,str):
        edges_color_real_reported = edges_color
        edges_color_only_real = edges_color
        edges_color_only_reported = edges_color
    else:
        edges_color_real_reported,edges_color_only_real,edges_color_only_reported = edges_color

    if reported_cmatrix is None:
        reported_cmatrix= real_cmatrix

    for idx1,idx2 in product(range(l),range(l)):
        if idx1==idx2:
            continue
        real_edge = real_cmatrix[idx1,idx2] and real_cmatrix[idx2,idx1]
        reported_edge = reported_cmatrix[idx1,idx2] and reported_cmatrix[idx2,idx1]
        edge_color = edges_color_real_reported if (real_edge and reported_edge) else (edges_color_only_reported if reported_edge else edges_color_only_real)
        if real_edge or reported_edge:
            x1 = nodes[idx1].x
            y1 = nodes[idx1].y
            x2 = nodes[idx2].x
            y2 = nodes[idx2].y
            yield [x1,x2],[y1,y2],edge_color

def visualize_cmatrix(nodes:list[rn.Node],real_cmatrix:np.ndarray,fig,edges_color = ('black','red','orange'),reported_cmatrix:np.ndarray=None,is_plotly=True):
    
    for x,y,edge_color in enumerate_edges_to_display(nodes,real_cmatrix,fig,edges_color,reported_cmatrix):
        if is_plotly:
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color=edge_color),showlegend=False))
        else:
            ax = fig.get_axes()[0]

            ax.plot(x, y, linestyle='-',marker='',color=edge_color)




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
    return fig