# RF connectivity simulator


https://github.com/Galaz7/RF_Connectivity_Simulator/assets/94379850/569b9407-217d-4e61-be04-ad9b2e0bf2c0



Given a predifined area size, it randomizes location of nodes + velocity.
On every step , the simulator calculates the connectivity map, and reported connectivity according to report period that is randomized in poisson process with a given rate (e.g 3mins/15mins)

The output of the simulator:
- Accuracy
- Precision/Recall
- Islands connectivity
- Connectivity video

see [network simulator](notebooks/network_simulator.ipynb) for example



