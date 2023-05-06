# RL_TL
Reinforcement learning based control of Traffic Lights at crossroads junction simulated in SUMO. Here 2 Traffic lights at opposite junctions are simulated and interactions between them is shaped by the decisions made by individual agents.

Currently uses Deep Q-learning to train both agents individually. The state for both agents are given by an 'occupancy map' of vehicles in corresponding 'cells' that are divided based on distance from the Traffic light. Each vehicle coming in from an incoming lane of a Traffic Light is assigned a different cell number based on the lane and the distance and this is used to modify the occupancy map. Each Light has two actions which are switching the Red signal between alternate pair of opposing roads. Each road has 2 lanes for travel in both directions, up and down.

The environment for the learning and simulating process are enabled by the [SUMO](https://www.eclipse.org/sumo/) (Simulation of Urban Mobility) software and its python API [traci](https://sumo.dlr.de/docs/TraCI.html). The env consisting of the road system and traffic lights are represented by the [.net.xml](https://github.com/TheMedicineSeller/RL_TL/blob/master/sumo_files/handmade.net.xml) file that designates the lanes, roads, junctions with ids and a [.rou.xml](https://github.com/TheMedicineSeller/RL_TL/blob/master/sumo_files/handmade.rou.xml) file describing the routes of each of the cars that travel across.

The performance is judged by inferring the average wait time and the average queue length for all cars. It is also judged by running the simulation of trained agents.

# Prerqusites
1. SUMO api, both [cli and gui](https://sumo.dlr.de/docs/Installing/index.html)
2. traci python library (pip install traci)
3. tensorflow (for loading models and for training) 

# Future improvements
Implement Multi agent DDPG to better optimize training both agents into developing a shared concern for the wait time incurred by both.
