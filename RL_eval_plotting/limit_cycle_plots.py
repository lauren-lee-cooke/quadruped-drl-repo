# Plots for plotting limit cycles of the system
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

class PlotLimitCycles():
    
    def __init__(self, csv_file_path = any, save_path = any):
        
        df = pd.read_csv(csv_file_path)
        self.full_data = df
        self.save_path = save_path
        
        indices = []
    
        # separate data into different episodes 
        for index, row in self.full_data.iterrows():
            if row[0] == 0 and index != 0: 
                indices.append(index)
        indices.append(self.full_data.index[-1])
        
        self.num_episode = len(indices)
        self.epiosde_data = []
        self.foot_contact_data = []
        
         
        for i in range(len(indices) - 1):
            start = indices[i]
            end = indices[i+1]
            df_temp = self.full_data.iloc[start:end]
            self.epiosde_data.append(df_temp)
        
        for j in range(self.num_episode - 1):
            self.episode_save_path = "Episode " + str(j) + "/"
            os.makedirs((self.save_path + self.episode_save_path + "Limit Cycles/"), exist_ok=True)
            os.makedirs((self.save_path + self.episode_save_path + "Limit Cycles/Leg Plots/"), exist_ok=True)
            os.makedirs((self.save_path + self.episode_save_path + "Limit Cycles/Leg Plots/BL"), exist_ok=True)
            os.makedirs((self.save_path + self.episode_save_path + "Limit Cycles/Leg Plots/BR"), exist_ok=True)
            os.makedirs((self.save_path + self.episode_save_path + "Limit Cycles/Leg Plots/FR"), exist_ok=True)
            os.makedirs((self.save_path + self.episode_save_path + "Limit Cycles/Leg Plots/FL"), exist_ok=True)
            os.makedirs((self.save_path + self.episode_save_path + "Limit Cycles/Body Plots/"), exist_ok=True)
            os.makedirs((self.save_path + self.episode_save_path + "Limit Cycles/Leg Comparison Plots/"), exist_ok=True)
            
        
        self.plot_bodyRollvsPitch()
        self.plot_footZvsX()
        self.plot_BodyQvsQdot()
        self.plot_LegQvsQdot()
        self.plot_jointQvsQdot()
        self.plot_Body3D()
        self.plot_feet3D()
        
    
    def plot_bodyRollvsPitch(self):
        

        for i in range(self.num_episode - 1):
            
            plt.close()
            x = self.changeRadtoDegrees(self.epiosde_data[i]["Body Pitch"])
            y = self.changeRadtoDegrees(self.epiosde_data[i]["Body Roll"])
            
            plt.plot(x, y, color = "r", label="Body Pitch")
            fig = plt.plot(x.values[0], y.values[0], "bs", markersize=4)
            plt.xlabel("Pitch $(^\circ)$")
            plt.ylabel("Roll $(^\circ)$")
            plt.grid()
            plt.title(("Body Pitch vs Roll"))
            plt.legend(fig[0:], ["Initial Position"], loc="upper right", frameon = False)
            plt.savefig(self.save_path + "Episode " + str(i) + "/" + "Limit Cycles/" + "00a Roll vs Pitch" + ".png")
            plt.clf()
            
            # ============================== 3D PLOT =================================
            z = self.epiosde_data[i]["Body x"]
            
            fig = plt.figure(figsize=(13, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            ax.plot(z, x, y, linewidth=2.5)
            
            ax.set_ylabel("Pitch $(^\circ)$", fontsize=20)
            ax.set_zlabel("Roll $(^\circ)$", fontsize=20)
            ax.xaxis.set_ticklabels([])
            
            # All used to adjust axis to how you want it to look 
            ax.set_box_aspect((4, 1, 1))
            #ax.view_init(elev=20, azim =65 )
            #ax.dist = 8
            #ax.set_xlim(max(z), min(z))
            # Alternatively:
            ax.view_init(elev=None, azim =None)
            ax.dist = 8
            
            plt.title(("Body Pitch vs Roll"), fontsize=20)
            plt.savefig(self.save_path + "Episode " + str(i) + "/" + "Limit Cycles/" + "00b Roll vs Pitch" + ".png")
            plt.clf()
    
    
    def plot_footZvsX(self):
        
        for i in range(self.num_episode - 1):
            
            plt.close()
            x1 = self.epiosde_data[i]["FR Foot Position X"]
            y1 = self.epiosde_data[i]["FR Foot Position Z"]
            
            x2 = self.epiosde_data[i]["FL Foot Position X"]
            y2 = self.epiosde_data[i]["FL Foot Position Z"]
            
            x3 = self.epiosde_data[i]["BR Foot Position X"]
            y3 = self.epiosde_data[i]["BR Foot Position Z"]
            
            x4 = self.epiosde_data[i]["BL Foot Position X"]
            y4 = self.epiosde_data[i]["BL Foot Position Z"]
            
            fig, axs = plt.subplots(4, sharex=True, sharey = True)
            axs[0].plot(x1, y1, color = "C0", linewidth=2)
            axs[0].set_title("Foot Z vs X")
            axs[0].grid()
            axs[0].tick_params(axis='x', which='both', bottom=False)
            
            axs[1].plot(x2, y2, color = "C1", linewidth=2)
            axs[1].grid()
            axs[1].tick_params(axis='x', which='both', bottom=False)
            #axs[1].set_title("FL Foot Z vs X")
            
            axs[2].plot(x3, y3, color = "C2", linewidth=2)
            axs[2].grid()
            axs[2].tick_params(axis='x', which='both', bottom=False)
            #axs[2].set_title("BR Foot Z vs X")
            
            axs[3].plot(x4, y4, color = "C3", linewidth=2)
            axs[3].grid()
            #axs[3].set_title("BL Foot Z vs X")
            
            fig.legend(["FR", "FL", "BR", "BL"])
            
            fig.supxlabel("X (m)")
            fig.supylabel("Z (m)")
            plt.subplots_adjust(right = 0.85, top = 0.92, hspace=0.3)
            
            plt.savefig(self.save_path + "Episode " + str(i) + "/" + "Limit Cycles/" + "01 Foot Z vs X" + ".png")
            plt.clf()
    
    
    def plot_BodyQvsQdot(self):
        
        for i in range(self.num_episode - 1):
            
            plt.close()
            x1 = self.changeRadtoDegrees(self.epiosde_data[i]["Body Roll"])
            x2 = self.changeRadtoDegrees(self.epiosde_data[i]["Body Pitch"])
            x3 = self.changeRadtoDegrees(self.epiosde_data[i]["Body Yaw"])
            y1 = self.changeRadtoDegrees(self.epiosde_data[i]["Body Ang Vel x"])
            y2 = self.changeRadtoDegrees(self.epiosde_data[i]["Body Ang Vel y"])
            y3 = self.changeRadtoDegrees(self.epiosde_data[i]["Body Ang Vel z"])
            
            plt.plot(x1, y1, color = "b", label="Roll", alpha=0.7)
            plt.plot(x2, y2, color = "r", label="Pitch", alpha=0.7)
            # Yaw not plotted as it should not create a limit cycle
            
            plt.xlabel(r"$\theta  \: (^\circ)$")
            plt.ylabel(r"$\dot{\theta}  \: (^\circ / \,s)$")
            plt.grid()
            plt.title(("Body " + r"$\theta$ " +" vs " + r"$\dot{\theta}$"))  
    
            plt.legend()
            plt.savefig(self.save_path + "Episode " + str(i) + "/" + "Limit Cycles/Body Plots/" + "02 Body Q vs Q_dot Full" + ".png")
            plt.clf()
            
            
            # ============================== 3D PLOT =================================
            z = self.epiosde_data[i]["Body x"]
            
            fig = plt.figure(figsize=(13, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            ax.plot(z, x1, y1, linewidth=2.5, label = "Roll")
            ax.plot(z, x2, y2, linewidth=2.5, label = "Pitch")
            
            ax.set_ylabel(r"$\theta  \: (^\circ)$", fontsize=20)
            ax.set_zlabel(r"$\dot{\theta}  \: (^\circ / \,s)$", fontsize=20)
            ax.xaxis.set_ticklabels([])
            
            # All used to adjust axis to how you want it to look 
            ax.set_box_aspect((4, 1, 1))
            #ax.view_init(elev=20, azim =65 )
            #ax.dist = 8
            #ax.set_xlim(max(z), min(z))
            # Alternatively:
            ax.view_init(elev=None, azim =None)
            ax.dist = 7
            
            plt.title(("Body " + r"$\theta$ " +" vs " + r"$\dot{\theta}$"), fontsize=20)
            plt.legend()
            plt.savefig(self.save_path + "Episode " + str(i) + "/" + "Limit Cycles/Body Plots/" + "02a Body Q vs Q_dot Full" + ".png")
            plt.clf()
            
        
        for i in range(self.num_episode - 1):
            
            plt.close()
            x1 = self.changeRadtoDegrees(self.epiosde_data[i]["Body Roll"])
            y1 = self.changeRadtoDegrees(self.epiosde_data[i]["Body Ang Vel x"])
            
            plt.plot(x1, y1, label="Body Roll")
            fig1 = plt.plot(x1.values[0], y1.values[0], "rs", markersize=4)
            
            plt.xlabel(r"$\theta  \: (^\circ)$")
            plt.ylabel(r"$\dot{\theta}  \: (^\circ / \,s)$")
            plt.grid()
            plt.title(("Body X-Axis " + r"$\theta$ " +" vs " + r"$\dot{\theta}$"))  

            plt.legend(fig1[0:], ["Initial Position"], loc="upper right", frameon = False)
            plt.savefig(self.save_path + "Episode " + str(i) + "/" + "Limit Cycles/Body Plots/" + "02 Body Q vs Q_dot Roll" + ".png")
            plt.clf()
            
            
            # ============================== 3D PLOT =================================
            z = self.epiosde_data[i]["Body x"]
            
            fig = plt.figure(figsize=(13, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            ax.plot(z, x1, y1, linewidth=2.5, label = "Roll")
            
            ax.set_ylabel(r"$\theta  \: (^\circ)$", fontsize=20)
            ax.set_zlabel(r"$\dot{\theta}  \: (^\circ / \,s)$", fontsize=20)
            ax.xaxis.set_ticklabels([])
            
            # All used to adjust axis to how you want it to look 
            ax.set_box_aspect((4, 1, 1))
            #ax.view_init(elev=20, azim =65 )
            #ax.dist = 8
            #ax.set_xlim(max(z), min(z))
            # Alternatively:
            ax.view_init(elev=None, azim =None)
            ax.dist = 7
            
            plt.title(("Body Roll" + r"$\theta$ " +" vs " + r"$\dot{\theta}$"), fontsize=20)
            plt.savefig(self.save_path + "Episode " + str(i) + "/" + "Limit Cycles/Body Plots/" + "02b Body Q vs Q_dot Roll" + ".png")
            plt.clf()
        
        for i in range(self.num_episode - 1):
            
            plt.close()
            x2 = self.changeRadtoDegrees(self.epiosde_data[i]["Body Pitch"])
            y2 = self.changeRadtoDegrees(self.epiosde_data[i]["Body Ang Vel y"])
            
            plt.plot(x2, y2, label="Body Pitch")
            fig1 = plt.plot(x1.values[0], y1.values[0], "rs", markersize=4)
            
            plt.xlabel(r"$\theta  \: (^\circ)$")
            plt.ylabel(r"$\dot{\theta}  \: (^\circ / \, s)$")
            plt.grid()
            plt.title(("Body Y-Axis " + r"$\theta$ " +" vs " + r"$\dot{\theta}$")) 

            plt.legend(fig1[0:], ["Initial Position"], loc="upper right", frameon = False)
            plt.savefig(self.save_path + "Episode " + str(i) + "/" + "Limit Cycles/Body Plots/" + "02 Body Q vs Q_dot Pitch" + ".png")
            plt.clf()
            
            
            # ============================== 3D PLOT =================================
            z = self.epiosde_data[i]["Body x"]
            
            fig = plt.figure(figsize=(13, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            ax.plot(z, x2, y2, linewidth=2.5, label = "Pitch")
            
            ax.set_ylabel(r"$\theta  \: (^\circ)$", fontsize=20)
            ax.set_zlabel(r"$\dot{\theta}  \: (^\circ / \,s)$", fontsize=20)
            ax.xaxis.set_ticklabels([])
            
            # All used to adjust axis to how you want it to look 
            ax.set_box_aspect((4, 1, 1))
            #ax.view_init(elev=20, azim =65 )
            #ax.dist = 8
            #ax.set_xlim(max(z), min(z))
            # Alternatively:
            ax.view_init(elev=None, azim =None)
            ax.dist = 7
            
            plt.title(("Body Pitch" + r"$\theta$ " +" vs " + r"$\dot{\theta}$"), fontsize=20)
            plt.savefig(self.save_path + "Episode " + str(i) + "/" + "Limit Cycles/Body Plots/" + "02c Body Q vs Q_dot Pitch" + ".png")
            plt.clf()
        
        for i in range(self.num_episode - 1):
            
            plt.close()
            x3 = self.changeRadtoDegrees(self.epiosde_data[i]["Body Yaw"])
            y3 = self.changeRadtoDegrees(self.epiosde_data[i]["Body Ang Vel z"])
            
            plt.plot(x3, y3, label="Body Yaw")
            fig1 = plt.plot(x1.values[0], y1.values[0], "rs", markersize=4)
            
            plt.xlabel(r"$\theta  \: (^\circ)$")
            plt.ylabel(r"$\dot{\theta}  \: (^\circ / \, s)$")
            plt.grid()
            plt.title(("Body Z-Axis " + r"$\theta$ " +" vs " + r"$\dot{\theta}$")) 

            plt.legend(fig1[0:], ["Initial Position"], loc="upper right", frameon = False)
            plt.savefig(self.save_path + "Episode " + str(i) + "/" + "Limit Cycles/Body Plots/" + "02 Body Q vs Q_dot Yaw" + ".png")
            plt.clf()
            
            
            # ============================== 3D PLOT =================================
            z = self.epiosde_data[i]["Body x"]
            
            fig = plt.figure(figsize=(13, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            ax.plot(z, x3, y3, linewidth=2.5, label = "Yaw")
            
            ax.set_ylabel(r"$\theta  \: (^\circ)$", fontsize=20)
            ax.set_zlabel(r"$\dot{\theta}  \: (^\circ / \,s)$", fontsize=20)
            ax.xaxis.set_ticklabels([])
            
            # All used to adjust axis to how you want it to look 
            ax.set_box_aspect((4, 1, 1))
            #ax.view_init(elev=20, azim =65 )
            #ax.dist = 8
            #ax.set_xlim(max(z), min(z))
            # Alternatively:
            ax.view_init(elev=None, azim =None)
            ax.dist = 7
            
            plt.title(("Body Yaw" + r"$\theta$ " +" vs " + r"$\dot{\theta}$"), fontsize=20)
            
            plt.savefig(self.save_path + "Episode " + str(i) + "/" + "Limit Cycles/Body Plots/" + "02d Body Q vs Q_dot Yaw" + ".png")
            plt.clf()
    
    
    def plot_LegQvsQdot(self):
        
        for i in range(self.num_episode - 1):
            
            plt.close()
            x1 = self.changeRadtoDegrees(self.epiosde_data[i]["FR Hip Ad_Ab Angle"])
            y1 = self.changeRadtoDegrees(self.epiosde_data[i]["FR Hip Angle Vel"])
            x2 = self.changeRadtoDegrees(self.epiosde_data[i]["FL Hip Ad_Ab Angle"])
            y2 = self.changeRadtoDegrees(self.epiosde_data[i]["FL Hip Angle Vel"])
            x3 = self.changeRadtoDegrees(self.epiosde_data[i]["BR Hip Ad_Ab Angle"])
            y3 = self.changeRadtoDegrees(self.epiosde_data[i]["BR Hip Angle Vel"])
            x4 = self.changeRadtoDegrees(self.epiosde_data[i]["BL Hip Ad_Ab Angle"])
            y4 = self.changeRadtoDegrees(self.epiosde_data[i]["BL Hip Angle Vel"])
            
            plt.plot(x1, y1, color = "C0", label="FR", alpha=0.7)
            plt.plot(x2, y2, color = "C1", label="FL", alpha=0.7)
            plt.plot(x3, y3, color = "C2", label="BR", alpha=0.7)
            plt.plot(x4, y4, color = "C3", label="BL", alpha=0.7)
            
            plt.xlabel(r"$\theta  \: (^\circ)$")
            plt.ylabel(r"$\dot{\theta}  \: (^\circ / \, s)$")
            plt.grid()
            plt.title(("Hip Ad/Ab " + r"$\theta$ " +" vs " + r"$\dot{\theta}$"))  
            plt.legend()
            plt.savefig(self.save_path + "Episode " + str(i) + "/" + "Limit Cycles/Leg Comparison Plots/" + "03 Leg Q vs Q_dot Hip ad_ab" + ".png")
            plt.clf()
            
            # ============================== 3D PLOT =================================
            z = self.epiosde_data[i]["Body x"]
            
            fig = plt.figure(figsize=(13, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            ax.plot(z, x1, y1, linewidth=2.5, label = "FR")
            ax.plot(z, x2, y2, linewidth=2.5, label = "FL")
            ax.plot(z, x3, y3, linewidth=2.5, label = "BR")
            ax.plot(z, x4, y4, linewidth=2.5, label = "BL")
            
            ax.set_ylabel(r"$\theta  \: (^\circ)$", fontsize=20)
            ax.set_zlabel(r"$\dot{\theta}  \: (^\circ / \,s)$", fontsize=20)
            ax.xaxis.set_ticklabels([])
            
            # All used to adjust axis to how you want it to look 
            ax.set_box_aspect((4, 1, 1))
            #ax.view_init(elev=20, azim =65 )
            #ax.dist = 8
            #ax.set_xlim(max(z), min(z))
            # Alternatively:
            ax.view_init(elev=None, azim =None)
            ax.dist = 7
            
            plt.title(("Hip Ad/Ab " + r"$\theta$ " +" vs " + r"$\dot{\theta}$"), fontsize=20)
            plt.legend()
            plt.savefig(self.save_path + "Episode " + str(i) + "/" + "Limit Cycles/Leg Comparison Plots/" + "03a Leg Q vs Q_dot Hip ad_ab" + ".png")
            plt.clf()
        
        for i in range(self.num_episode - 1):
            
            plt.close()
            x1 = self.changeRadtoDegrees(self.epiosde_data[i]["FR Hip Swing Angle"])
            y1 = self.changeRadtoDegrees(self.epiosde_data[i]["FR Knee Angle Vel"])
            x2 = self.changeRadtoDegrees(self.epiosde_data[i]["FL Hip Swing Angle"])
            y2 = self.changeRadtoDegrees(self.epiosde_data[i]["FL Knee Angle Vel"])
            x3 = self.changeRadtoDegrees(self.epiosde_data[i]["BR Hip Swing Angle"])
            y3 = self.changeRadtoDegrees(self.epiosde_data[i]["BR Knee Angle Vel"])
            x4 = self.changeRadtoDegrees(self.epiosde_data[i]["BL Hip Swing Angle"])
            y4 = self.changeRadtoDegrees(self.epiosde_data[i]["BL Knee Angle Vel"])
            
            plt.plot(x1, y1, color = "C0", label="FR", alpha=0.7)
            plt.plot(x2, y2, color = "C1", label="FL", alpha=0.7)
            plt.plot(x3, y3, color = "C2", label="BR", alpha=0.7)
            plt.plot(x4, y4, color = "C3", label="BL", alpha=0.7)
            
            plt.xlabel(r"$\theta  \: (^\circ)$")
            plt.ylabel(r"$\dot{\theta}  \: (^\circ / \, s)$")
            plt.grid()
            plt.title(("Hip " + r"$\theta$ " +" vs " + r"$\dot{\theta}$")) 
            plt.legend()
            plt.savefig(self.save_path + "Episode " + str(i) + "/" + "Limit Cycles/Leg Comparison Plots/" + "03 Leg Q vs Q_dot Hip" + ".png")
            plt.clf()
            
            
            # ============================== 3D PLOT =================================
            z = self.epiosde_data[i]["Body x"]
            
            fig = plt.figure(figsize=(13, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            ax.plot(z, x1, y1, linewidth=2.5, label = "FR")
            ax.plot(z, x2, y2, linewidth=2.5, label = "FL")
            ax.plot(z, x3, y3, linewidth=2.5, label = "BR")
            ax.plot(z, x4, y4, linewidth=2.5, label = "BL")
            
            ax.set_ylabel(r"$\theta  \: (^\circ)$", fontsize=20)
            ax.set_zlabel(r"$\dot{\theta}  \: (^\circ / \,s)$", fontsize=20)
            ax.xaxis.set_ticklabels([])
            
            # All used to adjust axis to how you want it to look 
            ax.set_box_aspect((4, 1, 1))
            #ax.view_init(elev=20, azim =65 )
            #ax.dist = 8
            #ax.set_xlim(max(z), min(z))
            # Alternatively:
            ax.view_init(elev=None, azim =None)
            ax.dist = 7
            
            plt.title(("Hip " + r"$\theta$ " +" vs " + r"$\dot{\theta}$"), fontsize=20)
            plt.legend()
            plt.savefig(self.save_path + "Episode " + str(i) + "/" + "Limit Cycles/Leg Comparison Plots/" + "03b Leg Q vs Q_dot Hip" + ".png")
            plt.clf()
        
        for i in range(self.num_episode - 1):
            
            plt.close()
            x1 = self.changeRadtoDegrees(self.epiosde_data[i]["FR Knee Angle"])
            y1 = self.changeRadtoDegrees(self.epiosde_data[i]["FR Ankle Angle Vel"])
            x2 = self.changeRadtoDegrees(self.epiosde_data[i]["FL Knee Angle"])
            y2 = self.changeRadtoDegrees(self.epiosde_data[i]["FL Ankle Angle Vel"])
            x3 = self.changeRadtoDegrees(self.epiosde_data[i]["BR Knee Angle"])
            y3 = self.changeRadtoDegrees(self.epiosde_data[i]["BR Ankle Angle Vel"])
            x4 = self.changeRadtoDegrees(self.epiosde_data[i]["BL Knee Angle"])
            y4 = self.changeRadtoDegrees(self.epiosde_data[i]["BL Ankle Angle Vel"])
            
            plt.plot(x1, y1, color = "C0", label="FR", alpha=0.7)
            plt.plot(x2, y2, color = "C1", label="FL", alpha=0.7)
            plt.plot(x3, y3, color = "C2", label="BR", alpha=0.7)
            plt.plot(x4, y4, color = "C3", label="BL", alpha=0.7)
            
            plt.xlabel(r"$\theta  \: (^\circ)$")
            plt.ylabel(r"$\dot{\theta}  \: (^\circ / \, s)$")
            plt.grid()
            plt.title(("Body " + r"$\theta$ " +" vs " + r"$\dot{\theta}$")) 
            plt.legend()
            plt.savefig(self.save_path + "Episode " + str(i) + "/" + "Limit Cycles/Leg Comparison Plots/" + "03 Leg Q vs Q_dot Knee" + ".png")
            plt.clf()
            
            
            # ============================== 3D PLOT =================================
            z = self.epiosde_data[i]["Body x"]
            
            fig = plt.figure(figsize=(13, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            ax.plot(z, x1, y1, linewidth=2.5, label = "FR")
            ax.plot(z, x2, y2, linewidth=2.5, label = "FL")
            ax.plot(z, x3, y3, linewidth=2.5, label = "BR")
            ax.plot(z, x4, y4, linewidth=2.5, label = "BL")
            
            ax.set_ylabel(r"$\theta  \: (^\circ)$", fontsize=20)
            ax.set_zlabel(r"$\dot{\theta}  \: (^\circ / \,s)$", fontsize=20)
            ax.xaxis.set_ticklabels([])
            
            # All used to adjust axis to how you want it to look 
            ax.set_box_aspect((4, 1, 1))
            #ax.view_init(elev=20, azim =65 )
            #ax.dist = 8
            #ax.set_xlim(max(z), min(z))
            # Alternatively:
            ax.view_init(elev=None, azim =None)
            ax.dist = 7
            
            plt.title(("Knee " + r"$\theta$ " +" vs " + r"$\dot{\theta}$"), fontsize=20)
            plt.legend()
            plt.savefig(self.save_path + "Episode " + str(i) + "/" + "Limit Cycles/Leg Comparison Plots/" + "03c Leg Q vs Q_Knee" + ".png")
            plt.clf()
    
    
    def plot_jointQvsQdot(self):
         
        x_index = ["FR Hip Ad_Ab Angle", "FL Hip Ad_Ab Angle", "BR Hip Ad_Ab Angle", "BL Hip Ad_Ab Angle", "FR Hip Swing Angle", "FL Hip Swing Angle", "BR Hip Swing Angle", "BL Hip Swing Angle", "FR Knee Angle", "FL Knee Angle", "BR Knee Angle", "BL Knee Angle"]
        
        y_index = ["FR Hip Angle Vel", "FL Hip Angle Vel", "BR Hip Angle Vel", "BL Hip Angle Vel", "FR Knee Angle Vel", "FL Knee Angle Vel", "BR Knee Angle Vel", "BL Knee Angle Vel", "FR Ankle Angle Vel", "FL Ankle Angle Vel", "BR Ankle Angle Vel", "BL Ankle Angle Vel"] 
        
        save_index = ["FR", "FL", "BR", "BL", "FR", "FL", "BR", "BL", "FR", "FL", "BR", "BL", "FR", "FL", "BR", "BL"]	
        
        colour_index = ["C0", "C1", "C2", "C3", "C0", "C1", "C2", "C3", "C0", "C1", "C2", "C3", "C0", "C1", "C2", "C3"]
        
        for j in range(len(x_index)):     
            for i in range(self.num_episode - 1):
                
                plt.close()
                x1 = self.changeRadtoDegrees(self.epiosde_data[i][x_index[j]])
                y1 = self.changeRadtoDegrees(self.epiosde_data[i][y_index[j]])
        
                plt.plot(x1, y1, color = colour_index[j])
                fig1 = plt.plot(x1.values[0], y1.values[0], "rs", markersize=4)
            
                plt.xlabel(r"$\theta  \: (^\circ)$")
                plt.ylabel(r"$\dot{\theta}  \: (^\circ / \, s)$")
                plt.grid()
                plt.title((x_index[j] + " " + r"$\theta$ " +" vs " + r"$\dot{\theta}$")) 
                
                plt.legend(fig1[0:], ["Initial Position"], loc="upper right")
                plt.savefig(self.save_path + "Episode " + str(i) + "/" + "Limit Cycles/Leg Plots/" + save_index[j] + "/" + x_index[j] + ".png")
                plt.clf()
               
                
            # ============================== 3D PLOT =================================
            z = self.epiosde_data[i]["Time"]
            
            fig = plt.figure(figsize=(13, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            ax.plot(z, x1, y1, linewidth=2.5, color = colour_index[j])
            
            ax.set_ylabel(r"$\theta  \: (^\circ)$", fontsize=20)
            ax.set_zlabel(r"$\dot{\theta}  \: (^\circ / \,s)$", fontsize=20)
            #ax.xaxis.set_ticklabels([])
            
            # All used to adjust axis to how you want it to look 
            ax.set_box_aspect((4, 1, 1))
            #ax.view_init(elev=20, azim =65 )
            #ax.dist = 8
            #ax.set_xlim(max(z), min(z))
            # Alternatively:
            ax.view_init(elev=None, azim =None)
            ax.dist = 7
            
            plt.title((x_index[j] + " " + r"$\theta$ " +" vs " + r"$\dot{\theta}$")) 
            plt.savefig(self.save_path + "Episode " + str(i) + "/" + "Limit Cycles/Leg Plots/" + save_index[j] + "/" + x_index[j] + "3D" + ".png")
    
    
    def plot_Body3D(self):
        
        for i in range(self.num_episode - 1):
            
            # 3D plots         
            plt.close()
            x1 = self.epiosde_data[i]["Body Lin Vel z"]
            y1 = self.epiosde_data[i]["Body z"]
            z1 = self.epiosde_data[i]["Body Lin Vel x"]
            
            ax = plt.figure().add_subplot(projection='3d')
            
            ax.plot(x1, y1, z1)
            
            ax.set_xlabel("Z Velocity (m/s)")
            ax.set_ylabel("Z Position (m)")
            ax.set_zlabel("X Velocity (m/s)")
            
            plt.savefig(self.save_path + "Episode " + str(i) + "/" + "Limit Cycles/04 3D PLot.png")
            plt.clf()

            # 2D plots of 3D params       
            plt.close()
            x1 = self.epiosde_data[i]["Body Lin Vel z"]
            y1 = self.epiosde_data[i]["Body z"]
            z1 = self.epiosde_data[i]["Body Lin Vel x"]
            
            fig, ax = plt.subplots(1, 2)
            
            ax[0].plot(x1, y1)
            ax[0].set_xlabel("Z Velocity (m/s)")
            ax[0].set_ylabel("Z Position (m)")
            
            ax[1].plot(x1, z1)
            ax[0].set_xlabel("Z Velocity (m/s)")
            ax[1].set_ylabel("X Velocity (m/s)")
            
            plt.subplots_adjust(wspace=0.35)
            plt.savefig(self.save_path + "Episode " + str(i) + "/" + "Limit Cycles/04 2D PLot of 3D params.png")
            plt.clf()
    
    
    def plot_feet3D(self):
        
        for i in range(self.num_episode - 1):
                        
            plt.close()
            x1 = self.epiosde_data[i]["FR Foot Lin Vel Z"]
            y1 = self.epiosde_data[i]["FR Foot Position Z"]
            z1 = self.epiosde_data[i]["FR Foot Lin Vel X"]
            
            ax = plt.figure().add_subplot(projection='3d')
            
            ax.plot(x1, y1, z1, color = "C0", alpha=0.7)
            ax.set_xlabel("Z Velocity (m/s)")
            ax.set_ylabel("Z Position (m)")
            ax.set_zlabel("X Velocity (m/s)")
            
            plt.savefig(self.save_path + "Episode " + str(i) + "/" + "Limit Cycles/05 FR foot 3D plot.png")
            plt.clf()
        
        
        for i in range(self.num_episode - 1):
                        
            plt.close()
            x1 = self.epiosde_data[i]["FL Foot Lin Vel Z"]
            y1 = self.epiosde_data[i]["FL Foot Position Z"]
            z1 = self.epiosde_data[i]["FL Foot Lin Vel X"]
            
            ax = plt.figure().add_subplot(projection='3d')
            
            ax.plot(x1, y1, z1, color = "C1", alpha=0.7)
            ax.set_xlabel("Z Velocity (m/s)")
            ax.set_ylabel("Z Position (m)")
            ax.set_zlabel("X Velocity (m/s)")
            
            plt.savefig(self.save_path + "Episode " + str(i) + "/" + "Limit Cycles/05 FL foot 3D plot.png")
            plt.clf()
       
        
        for i in range(self.num_episode - 1):
                        
            plt.close()
            x1 = self.epiosde_data[i]["BR Foot Lin Vel Z"]
            y1 = self.epiosde_data[i]["BR Foot Position Z"]
            z1 = self.epiosde_data[i]["BR Foot Lin Vel X"]
            
            ax = plt.figure().add_subplot(projection='3d')
            
            ax.plot(x1, y1, z1, color = "C2", alpha=0.7)
            ax.set_xlabel("Z Velocity (m/s)")
            ax.set_ylabel("Z Position (m)")
            ax.set_zlabel("X Velocity (m/s)")
            
            plt.savefig(self.save_path + "Episode " + str(i) + "/" + "Limit Cycles/05 BR foot 3D plot.png")
            plt.clf()
        
        
        for i in range(self.num_episode - 1):
                        
            plt.close()
            x1 = self.epiosde_data[i]["BL Foot Lin Vel Z"]
            y1 = self.epiosde_data[i]["BL Foot Position Z"]
            z1 = self.epiosde_data[i]["BL Foot Lin Vel X"]
            
            ax = plt.figure().add_subplot(projection='3d')
            
            ax.plot(x1, y1, z1, color = "C3", alpha=0.7)
            ax.set_xlabel("Z Velocity (m/s)")
            ax.set_ylabel("Z Position (m)")
            ax.set_zlabel("X Velocity (m/s)")
            
            plt.savefig(self.save_path + "Episode " + str(i) + "/" + "Limit Cycles/05 BL foot 3D plot.png")
            plt.clf()


    # Functions data processing 
    def changeRadtoDegrees(self, data):
        return data * 180 / np.pi
    
   
                