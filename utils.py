import configparser
from sumolib import checkBinary
import os
import sys
import matplotlib.pyplot as plt

def set_sumo(gui, sumocfg_file_name, max_steps):
    """
    Configure various parameters of SUMO
    """
    # sumo things - we need to import python modules from the $SUMO_HOME/tools directory
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    # setting the cmd mode or the visual mode    
    if gui == False:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
 
    # setting the cmd command to run sumo at simulation time
    sumo_cmd = [sumoBinary, "-c", os.path.join('sumo_files', sumocfg_file_name), "--no-step-log", "true", "--waiting-time-memory", str(max_steps)]

    return sumo_cmd

# def set_test_path(model_n, models_path_name='Models'):
#     """
#     Returns a model path that identifies the model number provided as argument and a newly created 'test' path
#     """
#     model_folder_path = os.path.join(os.getcwd(), models_path_name, 'session'+str(model_n), '')

#     if os.path.isdir(model_folder_path):    
#         return model_folder_path
#     else: 
#         sys.exit('The model number specified does not exist in the models folder')

def save_data_and_plot(data, filename, xlabel, ylabel, plot_folder="Plots"):
        """
        Produce a plot of performance of the agent over the session and save the relative data to txt
        """
        min_val = min(data)
        max_val = max(data)

        plt.rcParams.update({'font.size': 24})

        plt.plot(data)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.margins(0)
        plt.ylim(min_val - 0.05 * abs(min_val), max_val + 0.05 * abs(max_val))
        fig = plt.gcf()
        fig.set_size_inches(20, 11.25)
        fig.savefig(os.path.join(plot_folder, filename+'.jpg'), dpi=96)
        plt.close("all")

        # with open(os.path.join('Plots', 'plot_'+filename + '_data.txt'), "w") as file:
        #     for value in data:
        #             file.write("%s\n" % value)