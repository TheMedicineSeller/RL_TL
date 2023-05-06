import traci
import random
import timeit
import numpy as np
from create_traffic import gen_traffic
from utils import save_data_and_plot

# actions here are activations of each of the green phases
PHASE_NS_GREEN  = 0 # 00
PHASE_NS_YELLOW = 1
PHASE_EW_GREEN  = 2 # 01
PHASE_EW_YELLOW = 3

class TrainerSimulation:
    def __init__(self, Model1, Model2, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_states, num_actions, training_epochs):
        self._Model1 = Model1
        self._Model2 = Model2
        self._gamma = gamma
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions

        self._reward_store1 = []
        self._reward_store2 = []
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []
        self._training_epochs = training_epochs
        self._concern_factor = 0.25

    def get_state(self, TL):
        # Based on the TL, func formulates the state to feed into the network in the form of cell occupancy map and distance to each TL
        occupancy_map = np.zeros(self._num_states)
        car_list = traci.vehicle.getIDList()

        if TL == 1:
            proximal_lanes = ['STL_1_0', 'STL_1_1', 'WTL_1_0', 'WTL_1_1', 'NTL_1_0', 'NTL_1_1', 'TL_21_0', 'TL_21_1']
        elif TL == 2:
            proximal_lanes = ['TL_12_0', 'TL_12_1', 'NTL_2_0', 'NTL_2_1', 'ETL_2_0', 'ETL_2_1', 'STL_2_0', 'STL_2_1']

        for car in car_list:
            position_in_lane = traci.vehicle.getLanePosition(car)
            lane_id = traci.vehicle.getLaneID(car)
            # getting dist from TL
            TL_distance = 50 - position_in_lane
            if lane_id not in proximal_lanes:
                TL_distance += 50
            
            if lane_id == "STL_1_0" or lane_id == "STL_1_1":
                lane_group = 0
            elif lane_id == "WTL_1_0" or lane_id == "WTL_1_1":
                lane_group = 1
            elif lane_id == "NTL_1_0" or lane_id == "NTL_1_1":
                lane_group = 2
            elif lane_id == "TL_21_0" or lane_id == "TL_21_1":
                lane_group = 3
            elif lane_id == "TL_12_0" or lane_id == "TL_12_1":
                lane_group = 4
            elif lane_id == "NTL_2_0" or lane_id == "NTL_2_1":
                lane_group = 5
            elif lane_id == "ETL_2_0" or lane_id == "ETL_2_1":
                lane_group = 6
            elif lane_id == "STL_2_0" or lane_id == "STL_2_1":
                lane_group = 7
            else:
                lane_group = -1
            
            if TL_distance < 5:
                lane_cell = 0
            elif TL_distance < 10:
                lane_cell = 1
            elif TL_distance < 15:
                lane_cell = 2
            elif TL_distance < 20:
                lane_cell = 3
            elif TL_distance < 30:
                lane_cell = 4
            elif TL_distance < 40:
                lane_cell = 5
            elif TL_distance < 50:
                lane_cell = 6
            elif TL_distance < 65:
                lane_cell = 7
            elif TL_distance < 80:
                lane_cell = 8
            elif TL_distance <= 100:
                lane_cell = 9
            
            if lane_group >= 1 and lane_group <= 7:
                car_position = int(str(lane_group) + str(lane_cell))  # composition of the two postion ID to create a number in interval 0-79
                valid_car = True
            elif lane_group == 0:
                car_position = lane_cell
                valid_car = True
            else:
                valid_car = False  # outgoing cars not considered

            if valid_car:
                occupancy_map[car_position] = 1
        
        return occupancy_map
        
    
    def collect_waiting_times(self, TL):
        if TL == 1:
            incoming_roads = ["WTL_1", "NTL_1", "STL_1", "TL_21"]
        elif TL == 2:
            incoming_roads = ["ETL_2", "NTL_2", "STL_2", "TL_12"]
        
        car_list = traci.vehicle.getIDList()
        total_wait_time = 0
        for car_id in car_list:
            wt = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)
            if road_id in incoming_roads:
                    total_wait_time += wt

        return total_wait_time
    
    def choose_action(self, state, epsilon, TL):
        # Epsilon greedy policy of picking actions
        if random.random() < epsilon:
            return random.randint(0, self._num_actions - 1) # random action
        else:
            if TL == 1:
                return np.argmax(self._Model1.predict_one(state))
            elif TL == 2:
                return np.argmax(self._Model2.predict_one(state))
            
    def greedy_action(self, state, TL):
        if TL == 1:
            return np.argmax(self._Model1.predict_one(state))
        elif TL == 2:
            return np.argmax(self._Model2.predict_one(state))
            
    # Action denotes turning a given pair of TLs green, here last green activated pair is taken and made red/yellow
    def set_yellow_phase(self, old_action_number, TL_name):
        yellow_phase_code = old_action_number * 2 + 1
        traci.trafficlight.setPhase(TL_name, yellow_phase_code)
        
    # we  have only 2 actions for each TL , NS and EW
    def set_green_phase(self, action_number, TL_name):
        if action_number == 0:
            traci.trafficlight.setPhase(TL_name, PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase(TL_name, PHASE_EW_GREEN)
    
    def get_queue_length(self):
        # Get the number of cars with speed = 0 in every incoming edge
        incoming_roads = ["ETL_2", "NTL_2", "STL_2", "TL_12",
                          "WTL_1", "NTL_1", "STL_1", "TL_21"]
        queue_length = 0
        for road in incoming_roads:
            queue_length += traci.edge.getLastStepHaltingNumber(road)
        return queue_length

    def simulate(self, steps_todo):
        if (self._step + steps_todo) >= self._max_steps:
            steps_todo = self._max_steps - self._step
        # simulate steps_todo step in sumo
        while steps_todo > 0:
            traci.simulationStep()  
            self._step += 1
            steps_todo -= 1
            queue_length = self.get_queue_length()
            self._combined_queue_length += queue_length
            self._combined_waiting_time += queue_length
    
    # Experience replay training
    def replay(self):
        replay1 = self._Model1.get_samples(self._Model1.batch_size)
        replay2 = self._Model2.get_samples(self._Model2.batch_size)

        if len(replay1) > 0:
            states = np.array([val[0] for val in replay1])  # extract states from the batch
            next_states = np.array([val[3] for val in replay1])  # extract next states from the batch

            # prediction
            q_s_a = self._Model1.predict_batch(states)  # predict Q(state), for every sample
            q_s_a_d = self._Model1.predict_batch(next_states)  # predict Q(next_state), for every sample

            # setup training arrays
            x = np.zeros((len(replay1), self._num_states))
            y = np.zeros((len(replay1), self._num_actions))

            for i, b in enumerate(replay1):
                state, action, reward, _ = b[0], b[1], b[2], b[3]  # extract data from one sample
                current_q = q_s_a[i]  # get the Q(state) predicted before
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])  # update Q(state, action)
                x[i] = state
                y[i] = current_q  # Q(state) that includes the updated action value
            self._Model1.train_batch(x, y)
        
        if len(replay2) > 0:
            states = np.array([val[0] for val in replay2])
            next_states = np.array([val[3] for val in replay2])

            q_s_a = self._Model2.predict_batch(states)
            q_s_a_d = self._Model2.predict_batch(next_states)

            x = np.zeros((len(replay2), self._num_states))
            y = np.zeros((len(replay2), self._num_actions))

            for i, b in enumerate(replay2):
                state, action, reward, _ = b[0], b[1], b[2], b[3]
                current_q = q_s_a[i]
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])
                x[i] = state
                y[i] = current_q

            self._Model2.train_batch(x, y)

    def save_episode_stats(self):
        # Sum of negative rewards incurred by both the TLs
        # Sum of wait time of all the cars
        self._reward_store1.append(self._sum_neg_reward1)
        self._reward_store2.append(self._sum_neg_reward2)
        self._cumulative_wait_store.append(self._combined_waiting_time)
        self._avg_queue_length_store.append(self._combined_queue_length / self._max_steps)

    def run(self, episode, epsilon):
        
        start_time = timeit.default_timer()
        # first, generate rand routefile & set up sumo
        gen_traffic(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # inits
        # self._waiting_times = {}
        self._sum_neg_reward1 = 0
        self._sum_neg_reward2 = 0
        self._combined_queue_length = 0
        self._combined_waiting_time = 0
        old_total_wait1 = 0
        old_total_wait2 = 0
        old_state1 = -1
        old_state2 = -1
        old_action1 = -1
        old_action2 = -1
        
        self._step = 0
        while self._step < self._max_steps:

            # get current state of the intersection
            current_state1 = self.get_state(TL=1)
            current_state2 = self.get_state(TL=2)

            # calculate reward of previous action: (change in cumulative waiting time between actions)
            current_total_wait1 = self.collect_waiting_times(TL=1)
            current_total_wait2 = self.collect_waiting_times(TL=2)
            
            reward1 = old_total_wait1 - current_total_wait1
            reward2 = old_total_wait2 - current_total_wait2
            reward1 = reward1 + self._concern_factor * reward2
            reward2 = reward2 + self._concern_factor * reward1

            if self._step != 0:
                self._Model1.add_sample((old_state1, old_action1, reward1, current_state1))
                self._Model2.add_sample((old_state2, old_action2, reward2, current_state2))
            

            # choose the light phase to activate, based on the current state of the intersection
            action1 = self.choose_action(current_state1, epsilon, TL=1)
            action2 = self.choose_action(current_state2, epsilon, TL=2)

            # if the chosen phase(action) is different from the last phase, activate the yellow phase
            # action codes are sent to activate the right signal light
            if self._step != 0:
                y_triggered = False
                if old_action1 != action1:
                    self.set_yellow_phase(old_action1, "TL_1")
                    y_triggered = True
                if old_action2 != action2:
                    self.set_yellow_phase(old_action2, "TL_2")
                    y_triggered = True
                if y_triggered:
                    self.simulate(self._yellow_duration)

            # execute the phase selected before
            self.set_green_phase(action1, "TL_1")
            self.set_green_phase(action2, "TL_2")
            self.simulate(self._green_duration)

            # saving variables for later & accumulate reward
            old_state1 = current_state1
            old_state2 = current_state2
            old_action1 = action1
            old_action2 = action2
            old_total_wait1 = current_total_wait1
            old_total_wait2 = current_total_wait2

            if reward1 < 0:
                self._sum_neg_reward1 += reward1
            if reward2 < 0:
                self._sum_neg_reward2 += reward2

        self.save_episode_stats()
        print(f"At epsilon = {round(epsilon, 2)}")
        print(f"TL1 reward: {self._sum_neg_reward1}")
        print(f"TL2 reward: {self._sum_neg_reward2}")
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        print("Training...")
        start_time = timeit.default_timer()
        for _ in range(self._training_epochs):
            self.replay()
        training_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time, training_time
    
    def test_run(self, ep_seed, n_cars):

        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        gen_traffic(ep_seed, n_cars, self._max_steps)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        queue_length_episode = []
        TL1_rewards = []
        TL2_rewards = []
        step = 0
        old_total_wait1 = 0
        old_total_wait2 = 0
        old_action1 = -1
        old_action2 = -1

        while step < self._max_steps:
            current_state1 = self.get_state(1)
            current_state2 = self.get_state(2)
            current_total_wait1 = self.collect_waiting_times(1)
            current_total_wait2 = self.collect_waiting_times(2)
            reward1 = old_total_wait1 - current_total_wait1
            reward2 = old_total_wait2 - current_total_wait2

            action1 = self.greedy_action(current_state1, 1)
            action2 = self.greedy_action(current_state2, 2)

            if step != 0:
                y_triggered = False
                if old_action1 != action1:
                    self.set_yellow_phase(old_action1, "TL_1")
                    y_triggered = True
                if old_action2 != action2:
                    self.set_yellow_phase(old_action2, "TL_2")
                    y_triggered = True
                
                if y_triggered:
                    sim_steps = self._yellow_duration
                    if (self._step + sim_steps) >= self._max_steps:
                        sim_steps = self._max_steps - self._step
                    while sim_steps > 0:
                        traci.simulationStep()
                        step += 1
                        sim_steps -= 1
                        queue_length = self.get_queue_length()
                        queue_length_episode.append(queue_length)

            self.set_green_phase(action1, "TL_1")
            self.set_green_phase(action2, "TL_2")
            
            sim_steps = self._green_duration
            if (self._step + sim_steps) >= self._max_steps:
                sim_steps = self._max_steps - self._step
            while sim_steps > 0:
                traci.simulationStep()
                step += 1
                sim_steps -= 1
                queue_length = self.get_queue_length()
                queue_length_episode.append(queue_length)

            old_action1 = action1
            old_action2 = action2
            old_total_wait1 = current_total_wait1
            old_total_wait2 = current_total_wait2

            TL1_rewards.append(reward1)
            TL2_rewards.append(reward2)

        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)
        save_data_and_plot(queue_length_episode, "queue_length", "step", "queue length", plot_folder="Test_plot")
        save_data_and_plot(TL1_rewards, "Left_TL_reward",  "action", "reward", plot_folder="Test_plot")
        save_data_and_plot(TL2_rewards, "Right_TL_reward", "action", "reward", plot_folder="Test_plot")

        return simulation_time