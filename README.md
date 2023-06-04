# RF connectivity simulator

Given a predifined area size, it randomizes location of nodes + velocity.
On every step , the simulator calculates the connectivity map, and reported connectivity according to report period that is randomized in poisson process with a given rate (e.g 3mins/15mins)

The output of the simulator:
- Accuracy
- Precision/Recall
- Islands connectivity
- Connectivity video

see [network simulator](notebooks/network_simulator.ipynb) for example



## TODOs:

- [x] Add two types of nodes - each with different antenna height & velocity
- [x] Distributions of 4 clusters. Each cluster have a common velocity vector up to noise
- [x] Randomize 4 groups center of mass. In every group randomize according to some radius
- [x] Similar velocity for every group
- [x] Similar angle for all nodes
- [x] Show center of mass
- [x] Colorize types of nodes
- [x] Analyze why the output is so connected with the nominal sensitivity
- [x] Organize experiments - folders, yaml configurations, yaml results, description
- [x] Add randomness for the propogation model - x[dB]
- [X] Add Egli model - use it as a model
- [x] Add legend to the video
- [x] Fix the figure display axes ranges (to be centered)
- [x] Starting distribution should be such that there is a single segment of connection
- [x] Add measurements o

- [x] Increase number of executions
- [x] Motion model - do it circular
- [ ] Update motion model - take into account terrain height map

- [x] Define an edge as an edge that is stable in the last 2 min (e.g connected>70% time)
- [x] The reporting algorithm will report on stable connections

- [x] Colorize differnt clusters
- [ ] Add probability for packet loss - single number
- [ ] Fix longly rice model for less than 1 km --> egli
- [ ] Seperate between simulation video rendering rate and simulation rate

- [ ] How to know the right information for transmission
- [ ] Add node types in the legend


