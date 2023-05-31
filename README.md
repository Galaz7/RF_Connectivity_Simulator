# RF connectivity simulator

Given a predifined area size, it randomizes location of nodes + velocity.
On every step , the simulator calculates the connectivity map, and reported connectivity according to report period that is randomized in poisson process with a given rate (e.g 3mins/15mins)

The output of the simulator:
- Accuracy
- Connectivity video

see [network simulator](notebooks/network_simulator.ipynb) for example



## TODOs:

- [ ] Randomize 5 groups center of mass. In every group randomize according to some radius
- [x] Colorize types of nodes
- [ ] Analyze why the output is so connected with the nominal sensitivity
- [x] Add two types of nodes - each with different antenna height & velocity
- [ ] Distributions of 4 clusters. Each cluster have a common velocity vector up to noise
- [ ] Add randomness for the propogation model - x[dB]
- [ ] Add measurements of number of islands / number of reported islands - accuracy of segmentation
- [X] Add Egli model - use it as a model
- [ ] Add probability for packet loss - single number
- [x] Add legend to the video
- [ ] Fix longly rice model for less than 1 km --> egli
- [ ] Fix the figure display axes ranges (to be centered)
- [ ] Seperate between simulation video rendering rate and simulation rate

- [ ] Add node types in the legend
- [ ] Update radio propogation model - with terrain map
- [ ] Update motion model - take into account terrain height map


- [ ] Starting distribution should be such that there is a single segment of connection
- [ ] How to know the right information for transmission


