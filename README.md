# RF connectivity simulator

Given a predifined area size, it randomizes location of nodes + velocity.
On every step , the simulator calculates the connectivity map, and reported connectivity according to report period that is randomized in poisson process with a given rate (e.g 3mins/15mins)

The output of the simulator:
- Accuracy
- Precision/Recall
- Islands connectivity
- Connectivity video

see [network simulator](notebooks/network_simulator.ipynb) for example



