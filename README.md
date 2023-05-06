# RL_TL
Reinforcement learning based control of Traffic Lights at crossroads junction. Here 2 Traffic lights at opposite junctions are simulated and interactions between them is shaped by the decisions made by individual agents.
Currently uses Deep Q-learning to train both agents individually. The state for both agents are given by an 'occupancy map' of vehicles in corresponding 'cells' that are divided based on distance from the Traffic light. Each vehicle coming in from an incoming lane of a Traffic Light is assigned a different cell number based on the lane and the distance and this is used to modify the occupancy map. Each Light has two actions which are switching the Red signal between alternate pair of opposing roads. Each road has 2 lanes for travel in both directions, up and down.
The performance is judged by inferring the average wait time and the average queue length for all cars. It is also judged by running the simulation of trained agents.

# Future improvements
Implement Multi agent DDPG to better optimize training both agents into developing a shared concern for the wait time incurred by both.