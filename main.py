import os
import time #useful for slowing program to see printed info easier 

import Quadruped
from utils import SaveOnBestTrainingRewardCallback, plotRewards, plotTrainingData
from RL_eval_plotting.evaluation import convertDatatoCSV, PlotEvaluResults
from RL_eval_plotting.limit_cycle_plots import PlotLimitCycles
from RL_eval_plotting.eval_numbers import EvalModel_Ideal, EvalModel_Skewness, EvalModel_Symmetry, FootContactPlots

import pybullet as p
import gymnasium as gym 
import numpy as np
from array2gif import write_gif
from moviepy.editor import ImageSequenceClip
from moviepy.editor import *
from PIL import Image

from stable_baselines3 import PPO, SAC, TD3, A2C 
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.logger import configure


def main():

    # === Generic Specifications for Training ===
    log_number = "01_94" 
    algorithm_type = "PPO"
    train = True                                             # If false then it will just return the visuals of the specified training session 
    load_model, model_to_load_save_path = True, "0_quad_logs/quad_PPO_01_68/incremental_model_PPO_01_68/PPO01_68_1000000_steps.zip"
    training_steps, incremental_save_num = 2_000_000, 500_000 
    make_gif = True
    store_evaluation_data = True
    plot_evaluation_data = True
    plot_training_information = True
    model_to_render = None # should be None if you want to render the model that was trained in this script, otherwise specify the path to the model you want to render
    num_render_steps = 600
    render_quality = "high"
    render_terrain_type = "even" # "even", "uneven", "both",
    
    # For evaluating the model that already has data stored in a csv 
    eval_csv = False
    
    env_args = {"render_mode": "rgb_array", 
                "target_lin_vel_range": [1.0, 1.2], "friction_range": [0.8, 1],
                "r_velocity": 0.6, "r_energy": 0.05e-3, "r_alive": 0.6, "r_height": 0, "r_orientation": 2, 
                "Kp": 500, "Kd": 10,  "Kp_damping": 0, "turning": True} 
    
    model_hyperparams = {"learning_rate": 1e-4, "n_steps": 4096, "batch_size": 128, "n_epochs": 10, #5e-4
                                "gamma": 0.99, "gae_lambda": 0.95, "vf_coef": 1, "ent_coef": 1e-8,
                                "verbose": 1, "tensorboard_log": "1_tb_quad_logs/" + "Log " + algorithm_type + " " + log_number, 
                                "policy_kwargs": dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))}


    # ========= Creating a Directory for Saving ========= 
    # For Saving Model
    pic_gif_dir = "images/"
    log_name = "quad_" + algorithm_type + "_" + log_number
    log_dir = "0_quad_logs/" + log_name + "/"
    best_log_dir = log_dir + "incremental_model_" + algorithm_type + "_" + log_number + "/"
    save_path = os.path.join(log_dir, ("final_" + log_name))
    pics_save_path = os.path.join(log_dir + pic_gif_dir, (log_name + "_render_reward" + ".pdf"))
    train_save_path_t = os.path.join(log_dir + pic_gif_dir, (log_name + "_training_time" + ".pdf"))
    train_save_path_r = os.path.join(log_dir + pic_gif_dir, (log_name + "_training_reward" + ".pdf"))
    gif_save_path_even = os.path.join(log_dir + pic_gif_dir, (log_name + "_even_v3" + ".gif"))
    gif_save_path_uneven = os.path.join(log_dir + pic_gif_dir, (log_name + "_uneven_v3" + ".gif"))
    movie_save_path_even = os.path.join(log_dir + pic_gif_dir, (log_name + "_even_v3" + ".mp4"))
    movie_save_path_uneven = os.path.join(log_dir + pic_gif_dir, (log_name + "_uneven_v3" + ".mp4"))
    eval_dir_even = log_dir + "0 Even_terrain/eval_data/"
    eval_dir_uneven = log_dir + "0 Uneven_terrain/eval_data/"
    eval_number_dir = eval_dir_even + "eval_numbers/"
    eval_number_csv_dir = eval_dir_even + "eval_data.csv"

    # Ensure directories exist
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(best_log_dir, exist_ok=True)
    os.makedirs((log_dir + pic_gif_dir), exist_ok=True)
    os.makedirs(eval_dir_even, exist_ok=True)
    os.makedirs(eval_dir_uneven, exist_ok=True)
    
    #'''
    # ========= Making an Environment ========= 
    env = gym.make("SimpleQuad-v0.1", **env_args)
    env = Monitor(env, best_log_dir)  # Monitor allows saving of info to csv for plotting 

    # ========= Training Environment =========
    if train:
        match algorithm_type:
            case "TD3": 
                n_actions = env.action_space.shape[-1]
                action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

                if load_model:
                    if model_to_load_save_path is not None:
                        model = TD3.load(model_to_load_save_path, env=env,  device="cuda")
                    else: model = TD3.load(save_path, env=env)
                else: 
                    model_hyperparams.update({"action_noise": action_noise})
                    model = TD3("MlpPolicy", env, **model_hyperparams,  device="cuda")
                    
            case "PPO":
                if load_model: 
                    if model_to_load_save_path is not None:
                        model = PPO.load(model_to_load_save_path, env=env,  device="cuda")
                    else: model = PPO.load(save_path, env=env)
                else: 
                    model = PPO("MlpPolicy", env, **model_hyperparams,  device="cuda") 


        # Create log of all training info in a csv
        new_logger = configure(log_dir, ["csv"])
        model.set_logger(new_logger)

        # Creating Callbacks for incremental saving
        incremental_save_callback = CheckpointCallback( save_freq=incremental_save_num, save_path=best_log_dir, name_prefix=(algorithm_type + log_number))

        # Train the agent
        model.learn(total_timesteps=int(training_steps), callback=incremental_save_callback, tb_log_name=log_name, progress_bar=True)
        model.save(save_path)

        del model

    # ========= Preparing for Rendering Terrain ========= 
    if not eval_csv:
        eval_dir_paths = [eval_dir_uneven, eval_dir_even]
        gif_save_path = [gif_save_path_uneven, gif_save_path_even]
        movie_save_path = [movie_save_path_uneven, movie_save_path_even]
        uneven_terrain = [True, False]
        
        if render_terrain_type == "both":
            terrain_num = 2
        else:
            terrain_num =  1
            
        for m in range(terrain_num):
            
            match render_quality: 
                case "low":
                    fps = 100
                case "high":
                    fps = 100
                    env.close()
                    env_args["render_mode"] = "human"
                    env_args["render"] = True
                    env_args["make_gif"] = make_gif
                    env_args["plot_eval_data"] = store_evaluation_data
                    if render_terrain_type == "even":
                        env_args["uneven_terrain"] = uneven_terrain[1]
                    else: env_args["uneven_terrain"] = uneven_terrain[m]
                    env = gym.make("SimpleQuad-v0.1", **env_args)
            
            if model_to_render is None:
                match algorithm_type:
                    case "TD3":
                        model = TD3.load(save_path, env=env)
                    case "PPO":
                        model = PPO.load(save_path, env=env)
            else: 
                match algorithm_type:
                    case "TD3":
                        model = TD3.load(model_to_render, env=env)
                    case "PPO":
                        model = PPO.load(model_to_render, env=env)
                
                
            # ========= Rendering the training =========
            obs = model.env.reset()
            frames = [None]*num_render_steps
            
            i = 0

            for i in range(num_render_steps):  
                action, _ = model.predict(obs)
                obs, reward, done, truncated = model.env.step(action)
                image, info, col_names, eval_info = env.render()
                frames[i] = image
            
            # =========  Making a Gif =========
            if env_args["make_gif"]:
                clip = ImageSequenceClip(list(frames), fps=fps) # change fps to increase/decrease speed 
                if render_terrain_type == "even":
                    clip.write_gif(gif_save_path[1], fps=fps)
                    clip.write_videofile(movie_save_path[1])
                else:
                    clip.write_gif(gif_save_path[m], fps=fps)
                    clip.write_videofile(movie_save_path[m])
                    

            # =========  Plot reward ========= 
            if plot_training_information:
                info = np.array(info)
                plotRewards(info.tolist(), col_names, pics_save_path)
                plotTrainingData((log_dir + "progress.csv"), train_save_path_t, train_save_path_r)
            
            
            # =========  Plot rendered policy data ========= 
            if store_evaluation_data:   
                eval = np.array(eval_info)
                if render_terrain_type == "even":
                    convertDatatoCSV(eval, (eval_dir_paths[1] + "eval_data" + ".csv"))
                    eval = PlotEvaluResults((eval_dir_paths[1] + "eval_data.csv"), eval_dir_paths[1])
                    if plot_evaluation_data:
                        eval.plotEvalData()
                        PlotLimitCycles((eval_dir_paths[1] + "eval_data.csv"), eval_dir_paths[1])
                    
                    # =========  Plot eval numbers ========= 
                    EvalModel_Ideal(eval_number_csv_dir, eval_number_dir)
                    EvalModel_Skewness(eval_number_csv_dir, eval_number_dir)
                    EvalModel_Symmetry(eval_number_csv_dir, eval_number_dir)
                    
                else: 
                    convertDatatoCSV(eval, (eval_dir_paths[m] + "eval_data" + ".csv"))
                    eval = PlotEvaluResults((eval_dir_paths[m] + "eval_data.csv"), eval_dir_paths[m])
                    if plot_evaluation_data:
                        eval.plotEvalData()
                        PlotLimitCycles((eval_dir_paths[m] + "eval_data.csv"), eval_dir_paths[m])
                        
    else: 
        # =========  Plot eval numbers ========= 
        EvalModel_Ideal(eval_number_csv_dir, eval_number_dir)
        EvalModel_Skewness(eval_number_csv_dir, eval_number_dir)
        EvalModel_Symmetry(eval_number_csv_dir, eval_number_dir)    
        FootContactPlots(eval_number_csv_dir, eval_number_dir)
    
        
    env.close()
    

if __name__ == "__main__":
    main()