import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback


# === Store arrays for Plotting === 
class writingCSVs():
    """
    Class that allows you to store an array for later usage
    """
    def __init__(self):
        self.current_array = np.array([])
        self.full_array = None

    def arrayStorage(self, reward_array):        
      
        if self.full_array is None:
            self.full_array = np.array([])
            self.full_array = reward_array
        
        else:   
            self.current_array = reward_array
            self.full_array = np.vstack((self.full_array, self.current_array))

        array = self.full_array.tolist()

        return array


# === Plot Rewards Components vs Total Reward ===
def plotRewards(data, column_names, save_path):
    """
    Function that plots data using matplotlib and saves the plot to a file path 
    It takes in an array of data, matching column names for the data, as well as a save path for the final image 
    Ultimately it plots the data while returning nothing 
    """
     
    num = len(column_names) - 2
    plotData = [None]*(len(data))
    normPlotData = [None]*(len(data))
    rewardData = [None]*(len(data))
    doneData = [None]*(len(data))
    legend_names = [None]*(len(column_names)-1)
    

    for n in range(len(column_names)-1):
        legend_names[n] = column_names[n]

    # Plotting all of the reward trends over time 
    plt.close()
    
    for i in range(num + 1):

        for j in range(len(data)):
            plotData[j] = data[j][i]

            if j > 1:
                if (abs(data[j][num])  - abs(rewardData[j-1]) > 19): # This should be chaged to make it correclty record done condiions
                    doneData[j] = 1
                else: doneData[j] = 0

            rewardData[j] = data[j][num]
        
        
        plt.plot(range(len(data)), plotData)

    plt.legend(legend_names)
    plt.xlabel("Iterations")
    plt.ylabel("Reward and Reward Components")
    
    plt.savefig(save_path)
    plt.clf()


def plotTrainingData(csv_file_path, figure_save_path_time, figure_save_path_reward):
    """
    Function that plots data using matplotlib and saves the plot to a file path 
    It takes in a csv file path and the save path for the final plotted figure
    The data in the csv file path is generated using SB3, The data that is plotted 
    is the mean training time, the mean reward and these are plotted over each batched update
    All the data is plottted and then the file is saved to the file paths provided
    Ultimately the function plots the data while returning nothing 
    """
    
    df = pd.read_csv(csv_file_path)
          
    plt.close()
    plt.plot(df["rollout/ep_len_mean"], color = "r", label="Episode Mean Time")
    plt.xlabel("Batch Number")
    plt.title("Episode Mean Time")
    plt.savefig(figure_save_path_time)
    plt.clf()
    
    plt.close()
    plt.plot(df["rollout/ep_rew_mean"], color = "g", label="Episode Mean Reward")
    plt.xlabel("Batch Number")
    plt.title("Episode Mean Reward")
    plt.savefig(figure_save_path_reward)
    plt.clf()


# +++++ CALLBACK FUNCTIONS +++++

# === Logging Additional Params with Tensorboard ===
# This function is directly from Stable baselines 3 -https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html 
class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value = np.random.random()
        self.logger.record("random_value", value)
        return True


# === Save Best Training === 
# This function has been slgihtly modified from the original from Stable baselines 3 code -https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html 
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: How many steps until a check should be performed
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, mod_type: str, log_num: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.num_it = 0 
        self.mod_type = mod_type
        self.log_num = log_num
        self.log_dir = log_dir
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to ``env.step()``.

        For child callback (of an ``EventCallback``), this will be called
        when the event is triggered. -- In this case we are a class of "BaseCallback"

        :return: If the callback returns False, training is aborted early.
        """ 
        
        if self.n_calls % self.check_freq == 0:
          
            self.num_it = self.num_it + self.check_freq
            self.save_path = os.path.join(self.log_dir, ("model_num_" + str(self.mod_type) + "_" + str(self.log_num) + "_" + str(self.num_it)))

            print(f"Saving new best model to {self.save_path}")
            self.model.save(self.save_path)

        return True