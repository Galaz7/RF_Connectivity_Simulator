from itertools import product
import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass
import numpy as np
import plotly.graph_objects as go
import rf_network as rn

def visualize_nodes(nodes:list[rn.Node],fig,nodes_color):
    x_list = np.array([node.x for node in nodes])
    y_list = np.array([node.y for node in nodes])
    scatter_trace = go.Scatter(x=x_list, y=y_list, mode='markers',name='Nodes',marker_color=nodes_color)
    fig.add_trace(scatter_trace)


def visualize_cmatrix(nodes:list[rn.Node],cmatrix:np.ndarray,fig,edges_color):
    l = len(nodes)
    for idx1,idx2 in product(range(l),range(l)):
        if idx1==idx2:
            continue
        if cmatrix[idx1,idx2]:
            x1 = nodes[idx1].x
            y1 = nodes[idx1].y
            x2 = nodes[idx2].x
            y2 = nodes[idx2].y
            fig.add_trace(go.Scatter(x=[x1, x2], y=[y1, y2], mode='lines', line=dict(color=edges_color),showlegend=False))
            #break


def figure_visualize_nodes(nodes,fig=None,nodes_color='blue',edges_color='black'):
    if fig is None:
        fig = go.Figure()
    cmatrix = rn.create_connectivity_matrix(nodes)

    visualize_nodes(nodes,fig,nodes_color)
    visualize_cmatrix(nodes,cmatrix,fig,edges_color)
    # Update the layout
    fig.update_layout(
        title='Nodes',
        xaxis_title='km',
        yaxis_title='km',
        width=1000,
        height=1000

    )
    return fig