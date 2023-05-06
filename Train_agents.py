import os
import datetime
from shutil import copyfile

from Simulation import TrainerSimulation
from TLmodel import TLModel
from utils import set_sumo, save_data_and_plot

# from torch.utils.tensorboard import SummaryWriter

# Training settings
GUI = False
N_EPS = 100
MAX_STEPS = 4000
GREEN_DUR  = 10
YELLOW_DUR = 4

NUM_LAYERS = 4
LAYER_WIDTHS = [250, 350, 300, 100]
LR = 0.001
N_EPOCHS = 100
SAMPLING_BATCH_SIZE = 32
HISTORY_BUF_LEN = 50000

GAMMA = 0.9
EPSILON_MIN = 0.02
EPSILON_DECAY = 0.95
N_STATES  = 80
N_ACTIONS = 2

MODEL_UPDATE_FREQ = 50
SUMOCFG_FILE = "sumo_config.sumocfg"

if __name__ == "__main__":
    
    # config = import_train_configuration(config_file='training_settings.ini')
    # path = set_train_path('Models/')
    sumo_cmd = set_sumo(GUI, SUMOCFG_FILE, MAX_STEPS)

    ## IF TENSORBOARD
    # summary_writer = SummaryWriter('./logs/DualTrafficLight')
    # summary_writer.add_scalar('AvgReward', r_mean, global_step=step)
    

    Model1 = TLModel(
        NUM_LAYERS, 
        LAYER_WIDTHS, 
        SAMPLING_BATCH_SIZE, 
        LR, 
        N_STATES, 
        N_ACTIONS,
        HISTORY_BUF_LEN
    )

    Model2 = TLModel(
        NUM_LAYERS, 
        LAYER_WIDTHS, 
        SAMPLING_BATCH_SIZE, 
        LR, 
        N_STATES, 
        N_ACTIONS,
        HISTORY_BUF_LEN
    )
    print("Models created ...")
    folder_path = os.path.join(os.getcwd(), 'Models', '')
   
    Simulation = TrainerSimulation(
        Model1,
        Model2,
        sumo_cmd,
        GAMMA,
        MAX_STEPS,
        GREEN_DUR,
        YELLOW_DUR,
        N_STATES,
        N_ACTIONS,
        N_EPOCHS
    )
    
    episode = 0
    model_counter = 1
    timestamp_start = datetime.datetime.now()
    
    epsilon = 1.0
    while episode < N_EPS:
        print('\n----- Episode', str(episode+1), 'of', str(N_EPS))
        
        # Subtractive epsilon-greedy policy
        # epsilon = 1.0 - (episode / N_EPS)
        
        # Exponential decrease epsilon greedy policy
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        simulation_time, training_time = Simulation.run(episode, epsilon)  # run the simulation
        print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:', round(simulation_time+training_time, 1), 's')
        episode += 1

        if episode % MODEL_UPDATE_FREQ == 0:
            folder_path = os.path.join(os.getcwd(), 'Models', '')
            model_folder = os.path.join(folder_path, "session" + str(model_counter))
            Model1.save_model("TL1", model_folder)
            Model2.save_model("TL2", model_folder)
            model_counter += 1
            
            save_data_and_plot(Simulation._reward_store1, filename='Left_TL_concerned_reward',  xlabel='episode', ylabel='cum neg reward')
            save_data_and_plot(Simulation._reward_store2, filename='Right_TL_concerned_reward', xlabel='episode', ylabel='cum neg reward')


    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())

    save_data_and_plot(Simulation._cumulative_wait_store, filename='wait time', xlabel='episode', ylabel='cum waiting time (s)')
    save_data_and_plot(Simulation._avg_queue_length_store, filename='queue length', xlabel='episode', ylabel='Avg queue length (in vehicles)')