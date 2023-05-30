# RF connectivity simulator

Given a predifined area size, it randomizes location of nodes + velocity.
On every step , the simulator calculates the connectivity map, and reported connectivity according to report period that is randomized in poisson process with a given rate (e.g 3mins/15mins)

The output of the simulator:
- Accuracy
- Connectivity video

see [network simulator](notebooks/network_simulator.ipynb) for example



## TODOs:

- [ ] Update motion model - take into account terrain height map
- [ ] Update radio propogation model - with terrain map
- [ ] Starting distribution should be such that there is a single segment of connection
- [ ] Add legend to the video