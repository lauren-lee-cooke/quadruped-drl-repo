import os
import csv
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import simpson

# Value: I want the code to be able to give final deviation in terms of x -cords:
# Plot: deviation of y and z, but also standard devation of those coords (want to be low)
# Plot: Body pitch - is there a limit cycle?
# Value: standard deviation of body yaw and roll (want to be low)
# Plot: box plot of torques for each joint 
# Value: standard deviation of torques for each joint - similar?
# Value: How many times does a foot make contact with ground? (each foot)
# Plot: Length of time in air vs on ground - matplotlib event plot 
# Plot: hip angles (ad/ab) - limit cycle?
# Plot: hip angles (fl/ex) - limit cycle?
# Plot: knee angles - limit cycle?

def convertDatatoCSV(data, save_path):
    # We know we want columns: 
    # 1. Time, 2. Deviation in x, 3. Deviation in y, 4. Deviation in z, 5. Body Pitch, 6. Body Roll, 7. Body Yaw,  
    # 8. 12*Joint Positions, 9. 12*Torque, 10. 4*contact booleans 11. 12*foot positons 
    # Torque and joint positions are in the form: [FR, FL, BR, BL] for each joint
    
    headings = ["Time", "Body x", "Body y", "Body z", "Body Roll", "Body Pitch", "Body Yaw", 
                "Body Lin Vel x", "Body Lin Vel y", "Body Lin Vel z",
                "Body Ang Vel x", "Body Ang Vel y", "Body Ang Vel z",
                "FR Hip Ad_Ab Angle", "FR Hip Swing Angle", "FR Knee Angle",
                "FL Hip Ad_Ab Angle", "FL Hip Swing Angle", "FL Knee Angle",
                "BR Hip Ad_Ab Angle", "BR Hip Swing Angle", "BR Knee Angle",
                "BL Hip Ad_Ab Angle", "BL Hip Swing Angle", "BL Knee Angle",
                "FR Hip Angle Vel", "FR Knee Angle Vel", "FR Ankle Angle Vel",
                "FL Hip Angle Vel", "FL Knee Angle Vel", "FL Ankle Angle Vel",
                "BR Hip Angle Vel", "BR Knee Angle Vel", "BR Ankle Angle Vel",
                "BL Hip Angle Vel", "BL Knee Angle Vel", "BL Ankle Angle Vel",
                "FR Hip Torque", "FR Knee Torque", "FR Ankle Torque",
                "FL Hip Torque", "FL Knee Torque", "FL Ankle Torque",
                "BR Hip Torque", "BR Knee Torque", "BR Ankle Torque",
                "BL Hip Torque", "BL Knee Torque", "BL Ankle Torque", 
                "FR Contact", "FL Contact", "BR Contact", "BL Contact", 
                "FR GRF", "FL GRF", "BR GRF", "BL GRF",
                "FR Foot Position X", "FR Foot Position Y", "FR Foot Position Z", 
                "FL Foot Position X", "FL Foot Position Y", "FL Foot Position Z", 
                "BR Foot Position X", "BR Foot Position Y", "BR Foot Position Z", 
                "BL Foot Position X", "BL Foot Position Y", "BL Foot Position Z",
                "FR Foot Lin Vel X", "FR Foot Lin Vel Y", "FR Foot Lin Vel Z",
                "FL Foot Lin Vel X", "FL Foot Lin Vel Y", "FL Foot Lin Vel Z",
                "BR Foot Lin Vel X", "BR Foot Lin Vel Y", "BR Foot Lin Vel Z",
                "BL Foot Lin Vel X", "BL Foot Lin Vel Y", "BL Foot Lin Vel Z"]
    
    df = pd.DataFrame(data)
    df.to_csv(save_path, header= headings, index=False)

class PlotEvaluResults():
    """
    Class that plots data from a csv file
    """
    
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
        self.x_distance = []
        
        # Ranges of motion for each joint 
        self.FR_adab_range = []
        self.FR_hip_range = []
        self.FR_knee_range = []
        self.FL_adab_range = []
        self.FL_hip_range = []
        self.FL_knee_range = []
        self.BR_adab_range = []
        self.BR_hip_range = []
        self.BR_knee_range = []
        self.BL_adab_range = []
        self.BL_hip_range = []
        self.BL_knee_range = []
        self.FR_smoothed_contacts = []
        self.FL_smoothed_contacts = []
        self.BR_smoothed_contacts = []  
        self.BL_smoothed_contacts = []
        self.FR_smoothed_contacts.append([1,1])
        self.FL_smoothed_contacts.append([1,1])
        self.BR_smoothed_contacts.append([1,1])
        self.BL_smoothed_contacts.append([1,1])
 
        for i in range(len(indices) - 1):
            start = indices[i]
            end = indices[i+1]
            df_temp = self.full_data.iloc[start:end]
            self.epiosde_data.append(df_temp)
        
        
        for j in range(self.num_episode - 1):
            self.episode_save_path = "Episode " + str(j) + "/"
            os.makedirs((self.save_path + self.episode_save_path + "eval_plots/"), exist_ok=True)
            os.makedirs((self.save_path + self.episode_save_path + "eval_plots/Leg Angles/"), exist_ok=True)
            os.makedirs((self.save_path + self.episode_save_path + "eval_plots/Torques/"), exist_ok=True)
            os.makedirs((self.save_path + self.episode_save_path + "eval_plots/GRF/"), exist_ok=True)

        
        # Information for text file 
        self.lineBreak = "=============================="
    
    
    def plotEvalData(self):
        self.plotBodyDeviations()
        self.plotBodyAngles()
        self.plotJointAngles()
        self.plotJointAngles_separate()
        self.plotJointTorques()
        self.plotTorqueProfiles()
        self.plotFootContacts()
        self.footPositionPlot()
        self.writeSignificantInfotoFile()
        self.plotGroundReactionForces()
        #self.writeLegLoadingtoFile()

             
    def plotBodyDeviations(self): 
        # Value: I want the code to be able to give final deviation in terms of x -cords:
        # Plot: deviation of y and z, but also standard devation of those coords (want to be low)
        for i in range(self.num_episode - 1):
            
            x = self.epiosde_data[i]["Time"]
            y1 = self.epiosde_data[i]["Body y"]
            y2 = self.epiosde_data[i]["Body z"]
            
            self.x_distance.append(self.epiosde_data[i]["Body x"].values[-1])
            x_std = "Distance traveled: " + "{:.3f}".format((self.x_distance[i]))
            y1_std = "Y STD " + "{:.5f}".format(np.std(self.epiosde_data[i]["Body y"].values)) + " Mean" + "{:.5f}".format(np.mean(self.epiosde_data[i]["Body y"].values))
            y2_std = "Z STD " + "{:.3f}".format(np.std(self.epiosde_data[i]["Body z"].values)) + " Mean" + "{:.3f}".format(np.mean(self.epiosde_data[i]["Body z"].values))
            
            fig, axs = plt.subplots(2)
            axs[0].plot(x, y1)
            axs[0].set_title("Body-Y Deviations")
            axs[0].set_xlabel('Time (s)')
            axs[0].set_ylabel('Deviation (m)')
            x_lim = axs[0].get_xlim()
            ylim = axs[0].get_ylim()
            #axs[0].text(x_lim[1], ylim[1], y1_std, horizontalalignment="right", verticalalignment="top")
            
            axs[1].plot(x, y2, color = "g")
            axs[1].set_title("Body-Z Deviations")
            axs[1].set_xlabel('Time (s)')
            axs[1].set_ylabel('Deviation (m)')
            x_lim = axs[1].get_xlim()
            ylim = axs[1].get_ylim()
            #axs[1].text(x_lim[1], ylim[1], y2_std, horizontalalignment="right", verticalalignment="top")
            plt.subplots_adjust(left=0.15, hspace=0.8)
            
            #fig.text(0.5, 0.48, x_std, horizontalalignment="center", verticalalignment="center", **{"color": "m"})
            fig.align_ylabels(axs)
            fig.savefig(self.save_path + "Episode " + str(i) + "/eval_plots/" + "06 Body Deviations.png")
            plt.close()
    
    
    def plotBodyAngles(self):
        # Plot: Body pitch - is there a limit cycle?
        # Value: standard deviation of body yaw and roll (want to be low)
        
        for i in range(self.num_episode - 1):
            plt.close()
            x = self.epiosde_data[i]["Time"]
            y = self.epiosde_data[i]["Body Pitch"]
            y = self.changeRadtoDegrees(y)
            
            yaw_std = "Yaw std " + "{:.5f}".format(np.std(self.epiosde_data[i]["Body Yaw"]))
            roll_std = "Roll std " + "{:.5f}".format(np.std(self.epiosde_data[i]["Body Roll"]))
            text = yaw_std + "\n" + roll_std
            
            plt.plot(x, y, color = "r", label="Body Pitch")
            plt.xlabel("Time (s)")
            plt.ylabel("Angle $(^\circ)$")
            plt.title(("Body Pitch"))
            ax = plt.gca()
            x_lim = ax.get_xlim()
            ylim = ax.get_ylim()
            plt.text(0.98, 0.98, text, horizontalalignment="right", verticalalignment="top")
            plt.savefig(self.save_path + "Episode " + str(i) + "/eval_plots/" + "05 Body Pitch" + ".png")
            plt.clf()
            
            
    def plotJointAngles(self):
        # Plot: hip angles (ad/ab) - limit cycle?
        # Plot: hip angles (fl/ex) - limit cycle?
        # Plot: knee angles - limit cycle?
        for i in range(self.num_episode - 1):
                
            x = self.epiosde_data[i]["Time"]
            y1_1 = self.changeRadtoDegrees(self.epiosde_data[i]["FR Hip Ad_Ab Angle"])
            y1_2 = self.changeRadtoDegrees(self.epiosde_data[i]["FL Hip Ad_Ab Angle"])
            y1_3 = self.changeRadtoDegrees(self.epiosde_data[i]["BR Hip Ad_Ab Angle"])
            y1_4 = self.changeRadtoDegrees(self.epiosde_data[i]["BL Hip Ad_Ab Angle"])
            
            y2_1 = self.changeRadtoDegrees(self.epiosde_data[i]["FR Hip Swing Angle"])
            y2_2 = self.changeRadtoDegrees(self.epiosde_data[i]["FL Hip Swing Angle"])
            y2_3 = self.changeRadtoDegrees(self.epiosde_data[i]["BR Hip Swing Angle"])
            y2_4 = self.changeRadtoDegrees(self.epiosde_data[i]["BL Hip Swing Angle"])
            
            y3_1 = self.changeRadtoDegrees(self.epiosde_data[i]["FR Knee Angle"])
            y3_2 = self.changeRadtoDegrees(self.epiosde_data[i]["FL Knee Angle"])
            y3_3 = self.changeRadtoDegrees(self.epiosde_data[i]["BR Knee Angle"])
            y3_4 = self.changeRadtoDegrees(self.epiosde_data[i]["BL Knee Angle"])
            
            fig, axs = plt.subplots(3)
            axs[0].plot(x, y1_1)
            axs[0].plot(x, y1_2)
            axs[0].plot(x, y1_3)
            axs[0].plot(x, y1_4)
            axs[0].set_title("Hip Ad/AB Angle")
            axs[0].set_xlabel('Time')
            axs[0].set_ylabel('Angle $(^\circ)$')
            
            axs[1].plot(x, y2_1)
            axs[1].plot(x, y2_2)
            axs[1].plot(x, y2_3)
            axs[1].plot(x, y2_4)
            axs[1].set_title("Hip Swing Angles")
            axs[1].set_xlabel('Time')
            axs[1].set_ylabel('Angle $(^\circ)$')
            
            axs[2].plot(x, y3_1)
            axs[2].plot(x, y3_2)
            axs[2].plot(x, y3_3)
            axs[2].plot(x, y3_4)
            axs[2].set_title("Knee Angles")
            axs[2].set_xlabel('Time')
            axs[2].set_ylabel('Angle $(^\circ)$')
            
            plt.subplots_adjust(left=0.15, right=0.85, hspace=0.8)
            
            fig.legend(["FR", "FL", "BR", "BL"])
            fig.align_ylabels(axs)
            fig.savefig(self.save_path + "Episode " + str(i) + "/eval_plots/Leg Angles/" + "02 Leg Angles.png")
            plt.close()


    def plotJointAngles_separate(self):
        # Plot: hip angles (ad/ab) - limit cycle?
        # Plot: hip angles (fl/ex) - limit cycle?
        # Plot: knee angles - limit cycle?
        for i in range(self.num_episode - 1):
                
            x = self.epiosde_data[i]["Time"]
            y1_1 = self.changeRadtoDegrees(self.epiosde_data[i]["FR Hip Ad_Ab Angle"])
            y1_2 = self.changeRadtoDegrees(self.epiosde_data[i]["FL Hip Ad_Ab Angle"])
            y1_3 = self.changeRadtoDegrees(self.epiosde_data[i]["BR Hip Ad_Ab Angle"])
            y1_4 = self.changeRadtoDegrees(self.epiosde_data[i]["BL Hip Ad_Ab Angle"])
            
            y2_1 = self.changeRadtoDegrees(self.epiosde_data[i]["FR Hip Swing Angle"])
            y2_2 = self.changeRadtoDegrees(self.epiosde_data[i]["FL Hip Swing Angle"])
            y2_3 = self.changeRadtoDegrees(self.epiosde_data[i]["BR Hip Swing Angle"])
            y2_4 = self.changeRadtoDegrees(self.epiosde_data[i]["BL Hip Swing Angle"])
            
            y3_1 = self.changeRadtoDegrees(self.epiosde_data[i]["FR Knee Angle"])
            y3_2 = self.changeRadtoDegrees(self.epiosde_data[i]["FL Knee Angle"])
            y3_3 = self.changeRadtoDegrees(self.epiosde_data[i]["BR Knee Angle"])
            y3_4 = self.changeRadtoDegrees(self.epiosde_data[i]["BL Knee Angle"])
            
            plt.close()
            plt.plot(x, y1_1, color = "C0", label="FR")
            plt.plot(x, y1_2, color = "C1", label="FL")
            plt.plot(x, y1_3, color = "C2", label="BR")
            plt.plot(x, y1_4, color = "C3", label="BL")
            plt.xlabel("Time (s)")
            plt.ylabel("Angle $(^\circ)$")
            plt.legend()
            plt.title(("Hip Ad/AB Angle"))
            plt.savefig(self.save_path + "Episode " + str(i) + "/eval_plots/Leg Angles/" + "02a Leg Angles - Hip Ad Ab" + ".png")
            plt.clf()
            
            plt.plot(x, y2_1, color = "C0", label="FR")
            plt.plot(x, y2_2, color = "C1", label="FL")
            plt.plot(x, y2_3, color = "C2", label="BR")
            plt.plot(x, y2_4, color = "C3", label="BL")
            plt.xlabel("Time (s)")
            plt.ylabel("Angle $(^\circ)$")
            plt.legend()
            plt.title(("Hip Angle"))
            plt.savefig(self.save_path + "Episode " + str(i) + "/eval_plots/Leg Angles/" + "02b Leg Angles - Hip" + ".png")
            plt.clf()
            
            plt.plot(x, y3_1, color = "C0", label="FR")
            plt.plot(x, y3_2, color = "C1", label="FL")
            plt.plot(x, y3_3, color = "C2", label="BR")
            plt.plot(x, y3_4, color = "C3", label="BL")
            plt.xlabel("Time (s)")
            plt.ylabel("Angle $(^\circ)$")
            plt.legend()
            plt.title(("Knee Angle"))
            plt.savefig(self.save_path + "Episode " + str(i) + "/eval_plots/Leg Angles/" + "02c Leg Angles - Knee" + ".png")
            plt.clf()


    def plotJointTorques(self):
        # Plot: box plot of torques for each joint 
        # Value: standard deviation of torques for each joint - similar?
        
         
        for i in range(self.num_episode - 1):
            
            plt.close()
            x = self.epiosde_data[i]["Time"]
            y1 = self.epiosde_data[i]["FR Hip Torque"]
            y2 = self.epiosde_data[i]["FL Hip Torque"]
            y3 = self.epiosde_data[i]["BR Hip Torque"]
            y4 = self.epiosde_data[i]["BL Hip Torque"]
            
            fig, axs = plt.subplots(4, 3, sharex=True, sharey=True) 
            axs[0, 0].plot(x, y1, color = "C0")
            axs[1, 0].plot(x, y2, color = "C1")
            axs[2, 0].plot(x, y3, color = "C2")
            axs[3, 0].plot(x, y4, color = "C3")
            
            # Fixing ticks 
            axs[0, 0].tick_params(axis='x', which='both', bottom=False)
            axs[1, 0].tick_params(axis='x', which='both', bottom=False)
            axs[2, 0].tick_params(axis='x', which='both', bottom=False)
            
            axs[0, 0].set_title("Hip Ad/Ab Torque", fontsize=10)
            fig.legend(["FR", "FL", "BR", "BL"])
            
            x = self.epiosde_data[i]["Time"]
            y1 = self.epiosde_data[i]["FR Knee Torque"]
            y2 = self.epiosde_data[i]["FL Knee Torque"]
            y3 = self.epiosde_data[i]["BR Knee Torque"]
            y4 = self.epiosde_data[i]["BL Knee Torque"]
        
            axs[0, 1].plot(x, y1, color = "C0")
            axs[1, 1].plot(x, y2, color = "C1")
            axs[2, 1].plot(x, y3, color = "C2")
            axs[3, 1].plot(x, y4, color = "C3")
            
            axs[0, 1].set_title("Hip Torque", fontsize=10)
            
            # Fixing ticks 
            axs[0, 1].tick_params(axis='x', which='both', bottom=False)
            axs[1, 1].tick_params(axis='x', which='both', bottom=False)
            axs[2, 1].tick_params(axis='x', which='both', bottom=False)
            axs[0, 1].tick_params(axis='y', which='both', left=False)
            axs[1, 1].tick_params(axis='y', which='both', left=False)
            axs[2, 1].tick_params(axis='y', which='both', left=False)
            axs[3, 1].tick_params(axis='y', which='both', left=False)
            
            x = self.epiosde_data[i]["Time"]
            y1 = self.epiosde_data[i]["FR Ankle Torque"]
            y2 = self.epiosde_data[i]["FL Ankle Torque"]
            y3 = self.epiosde_data[i]["BR Ankle Torque"]
            y4 = self.epiosde_data[i]["BL Ankle Torque"]
            
            axs[0, 2].plot(x, y1, color = "C0")
            axs[1, 2].plot(x, y2, color = "C1")
            axs[2, 2].plot(x, y3, color = "C2")
            axs[3, 2].plot(x, y4, color = "C3")
            
            axs[0, 2].set_title("Knee Torque",  fontsize=10)
            
            # Fixing ticks 
            axs[0, 2].tick_params(axis='x', which='both', bottom=False)
            axs[1, 2].tick_params(axis='x', which='both', bottom=False)
            axs[2, 2].tick_params(axis='x', which='both', bottom=False)
            axs[0, 2].tick_params(axis='y', which='both', left=False)
            axs[1, 2].tick_params(axis='y', which='both', left=False)
            axs[2, 2].tick_params(axis='y', which='both', left=False)
            axs[3, 2].tick_params(axis='y', which='both', left=False)
            
            for n in range(4):
                for m in range(3):
                    axs[n, m].grid()
                    
            fig.supxlabel("Episode Time")
            fig.supylabel("Torque (N.m)")
            plt.subplots_adjust(right=0.85, hspace=0.4, wspace=0.4)
            plt.savefig(self.save_path + "Episode " + str(i) + "/eval_plots/Torques/" + "07 Torques.png")
            plt.clf()
            
            
    def plotTorqueProfiles(self):
        # Plot: box plot of torques for each joint 
        # Value: standard deviation of torques for each joint - similar?
        
         
        for i in range(self.num_episode - 1):
            
            plt.close()
            x = self.epiosde_data[i]["Time"]
            y1 = self.epiosde_data[i]["FR Hip Torque"]
            y2 = self.epiosde_data[i]["FL Hip Torque"]
            y3 = self.epiosde_data[i]["BR Hip Torque"]
            y4 = self.epiosde_data[i]["BL Hip Torque"]
            
            fig, axs = plt.subplots(4) 
            axs[0].plot(x, y1, color = "C0")
            axs[1].plot(x, y2, color = "C1")
            axs[2].plot(x, y3, color = "C2")
            axs[3].plot(x, y4, color = "C3")
            
            axs[0].grid()
            axs[1].grid()
            axs[2].grid()
            axs[3].grid()
            
            axs[0].tick_params(labelbottom = False, bottom = False)
            axs[1].tick_params(labelbottom = False, bottom = False)
            axs[2].tick_params(labelbottom = False, bottom = False)
        
            fig.supxlabel("Time (s)")
            fig.supylabel("Torque (Nm)")
            fig.suptitle(("Ad Ab Hip Torque"))
            fig.legend(["FR", "FL", "BR", "BL"])
            
            plt.subplots_adjust(right=0.85)
            plt.savefig(self.save_path + "Episode " + str(i) + "/eval_plots/Torques/" + "07a Ad Ab torque profiles.png")
            plt.clf()
            
            
            x = self.epiosde_data[i]["Time"]
            y1 = self.epiosde_data[i]["FR Knee Torque"]
            y2 = self.epiosde_data[i]["FL Knee Torque"]
            y3 = self.epiosde_data[i]["BR Knee Torque"]
            y4 = self.epiosde_data[i]["BL Knee Torque"]
            
            fig, axs = plt.subplots(4) 
            axs[0].plot(x, y1, color = "C0")
            axs[1].plot(x, y2, color = "C1")
            axs[2].plot(x, y3, color = "C2")
            axs[3].plot(x, y4, color = "C3")
            
            axs[0].grid()
            axs[1].grid()
            axs[2].grid()
            axs[3].grid()
            
            axs[0].tick_params(labelbottom = False, bottom = False)
            axs[1].tick_params(labelbottom = False, bottom = False)
            axs[2].tick_params(labelbottom = False, bottom = False)
            
            fig.supxlabel("Time (s)")
            fig.supylabel("Torque (Nm)")
            fig.suptitle(("Hip Torque"))
            fig.legend(["FR", "FL", "BR", "BL"])
            
            plt.subplots_adjust(right=0.85)
            plt.savefig(self.save_path + "Episode " + str(i) + "/eval_plots/Torques/" + "07b Hip torque profiles.png")
            plt.clf()
        
    
            x = self.epiosde_data[i]["Time"]
            y1 = self.epiosde_data[i]["FR Ankle Torque"]
            y2 = self.epiosde_data[i]["FL Ankle Torque"]
            y3 = self.epiosde_data[i]["BR Ankle Torque"]
            y4 = self.epiosde_data[i]["BL Ankle Torque"]
            
            fig, axs = plt.subplots(4) 
            axs[0].plot(x, y1, color = "C0")
            axs[1].plot(x, y2, color = "C1")
            axs[2].plot(x, y3, color = "C2")
            axs[3].plot(x, y4, color = "C3")

            axs[0].grid()
            axs[1].grid()
            axs[2].grid()
            axs[3].grid()
            
            axs[0].tick_params(labelbottom = False, bottom = False)
            axs[1].tick_params(labelbottom = False, bottom = False)
            axs[2].tick_params(labelbottom = False, bottom = False)
            
            fig.supxlabel("Time (s)")
            fig.supylabel("Torque (Nm)")
            fig.suptitle(("Knee Torque"))
            fig.legend(["FR", "FL", "BR", "BL"])
            
            plt.subplots_adjust(right=0.85)
            plt.savefig(self.save_path + "Episode " + str(i) + "/eval_plots/Torques/" + "07c Knee torque profiles.png")
            plt.clf()


    def plotFootContacts(self):
        # Value: How many times does a foot make contact with ground? (each foot)
        # Plot: Length of time in air vs on ground - matplotlib event plot 
        for i in range(self.num_episode - 1):
            plt.close()
        
            y1, y1_data = self.findContactNumber(self.epiosde_data[i]["FR Contact"].values)
            y2, y2_data = self.findContactNumber(self.epiosde_data[i]["FL Contact"].values)
            y3, y3_data = self.findContactNumber(self.epiosde_data[i]["BR Contact"].values)
            y4, y4_data = self.findContactNumber(self.epiosde_data[i]["BL Contact"].values)
            
            FR_contact = "FR Contact: " + str(y1)
            FL_contact = "FL Contact: " + str(y2)
            BR_contact = "BR Contact: " + str(y3)
            BL_contact = "BL Contact: " + str(y4)
            contact_text = FR_contact + " " + FL_contact + "\n" + BR_contact + " " + BL_contact
            
            testdict = {}
            testdict["FR"] = y1_data
            testdict["FL"] = y2_data
            testdict["BR"] = y3_data
            testdict["BL"] = y4_data
            
            fig, ax = plt.subplots()
            ax = self.plotbooleans(ax, testdict)
            
            self.epiosde_data[i]["FR Contact Smoothed"] = y1_data
            self.epiosde_data[i]["FL Contact Smoothed"] = y2_data
            self.epiosde_data[i]["BR Contact Smoothed"] = y3_data   
            self.epiosde_data[i]["BL Contact Smoothed"] = y4_data
            
            self.FR_smoothed_contacts.append(y1_data)
            self.FL_smoothed_contacts.append(y2_data)
            self.BR_smoothed_contacts.append(y3_data)
            self.BL_smoothed_contacts.append(y4_data)
            
            #plt.xlabel("Episode Time")
            plt.title(("Foot Contact Patterns"))
            
            ax = plt.gca()
            x_lim = ax.get_xlim()
            ylim = ax.get_ylim()
            plt.text(4.99, 4.99, contact_text)
            plt.subplots_adjust(bottom=0.2)
            plt.savefig(self.save_path + "Episode " + str(i) + "/eval_plots/" + "01 Foot Contacts.png")
            plt.clf()
    

    def footPositionPlot(self):
        # Deviation of feet in y and z positions
        
        for i in range(self.num_episode - 1):
            plt.close()
            x = self.epiosde_data[i]["Time"]
            y1 = self.epiosde_data[i]["FR Foot Position Y"]
            y2 = self.epiosde_data[i]["FL Foot Position Y"]
            y3 = self.epiosde_data[i]["BR Foot Position Y"]
            y4 = self.epiosde_data[i]["BL Foot Position Y"]
            
            fig, axs = plt.subplots(4, sharex=True, sharey=True) 
            
            axs[0].plot(x, y1, color = "C0")
            axs[1].plot(x, y2, color = "C1")
            axs[2].plot(x, y3, color = "C2")
            axs[3].plot(x, y4, color = "C3")
            
            fig.supxlabel("Episode Time")
            fig.supylabel("Deviations (m)")
            fig.suptitle(("Foot-Y Deviations"))
            fig.legend(["FR", "FL", "BR", "BL"])
            
            plt.subplots_adjust(right=0.85,  hspace=0.35)
            plt.savefig(self.save_path + "Episode " + str(i) + "/eval_plots/" + "04 Foot Y Deviations.png")
            plt.clf()
        
        for i in range(self.num_episode - 1):
            plt.close()
            x = self.epiosde_data[i]["Time"]
            y1 = self.epiosde_data[i]["FR Foot Position Z"]
            y2 = self.epiosde_data[i]["FL Foot Position Z"]
            y3 = self.epiosde_data[i]["BR Foot Position Z"]
            y4 = self.epiosde_data[i]["BL Foot Position Z"]
            
            fig, axs = plt.subplots(4, sharex=True, sharey=True) 
            axs[0].plot(x, y1, color = "C0")
            axs[1].plot(x, y2, color = "C1")
            axs[2].plot(x, y3, color = "C2")
            axs[3].plot(x, y4, color = "C3")
            
            fig.supxlabel("Time (s)")
            fig.supylabel("Deviations (m)")
            fig.suptitle(("Foot-Z Deviations"))
            fig.legend(["FR", "FL", "BR", "BL"])
            
            plt.subplots_adjust(right=0.85, hspace=0.35)
            plt.savefig(self.save_path + "Episode " + str(i) + "/eval_plots/" + "03 Foot Z Deviations.png")
            plt.clf()


    def plotGroundReactionForces(self):
        
        # Plot: GRF for each foot
        # Group image of GRF
        for i in range(self.num_episode - 1):
            plt.close()
            x = self.epiosde_data[i]["Time"].values
            y1 = self.epiosde_data[i]["FR GRF"]
            y2 = self.epiosde_data[i]["FL GRF"]
            y3 = self.epiosde_data[i]["BR GRF"]
            y4 = self.epiosde_data[i]["BL GRF"]
            
            c1 = self.epiosde_data[i]["FR Contact Smoothed"].values 
            c2 = self.epiosde_data[i]["FL Contact Smoothed"].values
            c3 = self.epiosde_data[i]["BR Contact Smoothed"].values 
            c4 = self.epiosde_data[i]["BL Contact Smoothed"].values
            
            fig, axs = plt.subplots(4, sharex=True, sharey=True) 
            axs[0].plot(x, y1, color = "C0")
            axs[0].margins(0, 0.2)
            axs[1].plot(x, y2, color = "C1")
            axs[2].plot(x, y3, color = "C2")
            axs[3].plot(x, y4, color = "C3")
            
            fig.legend(["FR", "FL", "BR", "BL"])
            
            j = 0
            for j in range(len(x) - 2):
                if j > 2:
                    if c1[j] != c1[j+1] and c1[j] == 0: # 0 -> 1
                        axs[0].axvline(x=x[j], color="r", linestyle=":", linewidth=1) # plot red line @ 0
                        axs[0].axvline(x=x[j + 1], color="k", linewidth=1) # plot black line @ 1
                    
                    elif c1[j] != c1[j+1] and c1[j] == 1: # 1 -> 0  
                        axs[0].axvline(x=x[j], color="k", linewidth=1)  # plot black line @ 1
                        axs[0].axvline(x=x[j + 1], color="r", linestyle=":", linewidth=1) # plot red line @ 0
                        
                
                    if c2[j] != c2[j+1] and c2[j] == 0: # 0 -> 1
                        axs[1].axvline(x=x[j], color="r", linestyle=":", linewidth=1) # plot red line @ 0
                        axs[1].axvline(x=x[j + 1], color="k", linewidth=1) # plot black line @ 1
                    
                    elif c2[j] != c2[j+1] and c2[j] == 1: # 1 -> 0  
                        axs[1].axvline(x=x[j], color="k", linewidth=1)  # plot black line @ 1
                        axs[1].axvline(x=x[j + 1], color="r", linestyle=":", linewidth=1) # plot red line @ 0
                        
                
                    if c3[j] != c3[j+1] and c3[j] == 0: # 0 -> 1
                        axs[2].axvline(x=x[j], color="r", linestyle=":", linewidth=1) # plot red line @ 0
                        axs[2].axvline(x=x[j + 1], color="k", linewidth=1) # plot black line @ 1
                    
                    elif c3[j] != c3[j+1] and c3[j] == 1: # 1 -> 0  
                        axs[2].axvline(x=x[j], color="k", linewidth=1)  # plot black line @ 1
                        axs[2].axvline(x=x[j + 1], color="r", linestyle=":", linewidth=1) # plot red line @ 0
                        
                
                    if c4[j] != c4[j+1] and c4[j] == 0: # 0 -> 1
                        axs[3].axvline(x=x[j], color="r", linestyle=":", linewidth=1) # plot red line @ 0
                        axs[3].axvline(x=x[j + 1], color="k", linewidth=1) # plot black line @ 1
                    
                    elif c4[j] != c4[j+1] and c4[j] == 1: # 1 -> 0  
                        axs[3].axvline(x=x[j], color="k", linewidth=1)  # plot black line @ 1
                        axs[3].axvline(x=x[j + 1], color="r", linestyle=":", linewidth=1) # plot red line @ 0
                        
            fig.supxlabel("Time (s)")
            fig.supylabel("Force (N)")
            fig.suptitle(("Ground Reaction Forces"))
            
            plt.subplots_adjust(right=0.85, hspace=0.35)
            plt.savefig(self.save_path + "Episode " + str(i) + "/eval_plots/GRF/" + "08 Ground Reaction Forces.png")
            plt.clf()
        
        # GRF for each foot idividually 
        for i in range(self.num_episode - 1):
            plt.close()
            x = self.epiosde_data[i]["Time"].values
            y1 = self.epiosde_data[i]["FR GRF"]
            y2 = self.epiosde_data[i]["FL GRF"]
            y3 = self.epiosde_data[i]["BR GRF"]
            y4 = self.epiosde_data[i]["BL GRF"]
            
            c1 = self.epiosde_data[i]["FR Contact Smoothed"].values 
            c2 = self.epiosde_data[i]["FL Contact Smoothed"].values
            c3 = self.epiosde_data[i]["BR Contact Smoothed"].values
            c4 = self.epiosde_data[i]["BL Contact Smoothed"].values 
            
            plt.figure(figsize=(15, 8))
            plt.plot(x, y1, color = "C0", linewidth = 3)
            plt.margins(0, 0.2)
            
            j = 0 
            for j in range(len(x) - 2):
                
                if j > 2:
                    if c1[j] != c1[j+1] and c1[j] == 0: # 0 -> 1
                        plt.axvline(x=x[j], color="r", linestyle=":", linewidth=1) # plot red line @ 0
                        plt.axvline(x=x[j + 1], color="k", linewidth=1) # plot black line @ 1
                    
                    elif c1[j] != c1[j+1] and c1[j] == 1: # 1 -> 0  
                        plt.axvline(x=x[j], color="k", linewidth=1)  # plot black line @ 1
                        plt.axvline(x=x[j + 1], color="r", linestyle=":", linewidth=1) # plot red line @ 0
                        
            plt.xlabel("Time (s)")
            plt.ylabel("Force (N)")
            plt.title(("FL Ground Reaction Forces"))
            
            plt.savefig(self.save_path + "Episode " + str(i) + "/eval_plots/GRF/" + "08a FR Ground Reaction Forces.png")
            plt.clf()
            
            plt.close()
            plt.figure(figsize=(15, 8))
            plt.plot(x, y2, color = "C1", linewidth = 3)
            plt.margins(0, 0.2)
    
            j = 0 
            for j in range(len(x) - 2):
                        
                if j > 2:
                    if c2[j] != c2[j+1] and c2[j] == 0: # 0 -> 1
                        plt.axvline(x=x[j], color="r", linestyle=":", linewidth=1) # plot red line @ 0
                        plt.axvline(x=x[j + 1], color="k", linewidth=1) # plot black line @ 1
                    
                    elif c2[j] != c2[j+1] and c2[j] == 1: # 1 -> 0  
                        plt.axvline(x=x[j], color="k", linewidth=1)  # plot black line @ 1
                        plt.axvline(x=x[j + 1], color="r", linestyle=":", linewidth=1) # plot red line @ 0
                        
                        
            plt.xlabel("Time (s)")
            plt.ylabel("Force (N)")
            plt.title(("FL Ground Reaction Forces"))
            
            plt.savefig(self.save_path + "Episode " + str(i) + "/eval_plots/GRF/" + "08b FL Ground Reaction Forces.png")
            plt.clf()
            
            plt.close()
            plt.figure(figsize=(15, 8))
            plt.plot(x, y3, color = "C2", linewidth = 3)
            plt.margins(0, 0.2)
            
            j = 0 
            for j in range(len(x) - 2):
                        
                if j > 2:
                    if c3[j] != c3[j+1] and c3[j] == 0: # 0 -> 1
                        plt.axvline(x=x[j], color="r", linestyle=":", linewidth=1) # plot red line @ 0
                        plt.axvline(x=x[j + 1], color="k", linewidth=1) # plot black line @ 1
                    
                    elif c3[j] != c3[j+1] and c3[j] == 1: # 1 -> 0  
                        plt.axvline(x=x[j], color="k", linewidth=1)  # plot black line @ 1
                        plt.axvline(x=x[j + 1], color="r", linestyle=":", linewidth=1) # plot red line @ 0
                        
            plt.xlabel("Time (s)")
            plt.ylabel("Force (N)")
            plt.title(("BR Ground Reaction Forces"))
            
            plt.savefig(self.save_path + "Episode " + str(i) + "/eval_plots/GRF/" + "08c BR Ground Reaction Forces.png")
            plt.clf()
            
            plt.close()
            plt.figure(figsize=(15, 8))
            plt.plot(x, y4, color = "C3", linewidth = 3)
            plt.margins(0, 0.2)
            
            j = 0 
            for j in range(len(x) - 2):

                if j > 2:
                    if c4[j] != c4[j+1] and c4[j] == 0: # 0 -> 1
                        plt.axvline(x=x[j], color="r", linestyle=":", linewidth=1) # plot red line @ 0
                        plt.axvline(x=x[j + 1], color="k", linewidth=1) # plot black line @ 1
                    
                    elif c4[j] != c4[j+1] and c4[j] == 1: # 1 -> 0  
                        plt.axvline(x=x[j], color="k", linewidth=1)  # plot black line @ 1
                        plt.axvline(x=x[j + 1], color="r", linestyle=":", linewidth=1) # plot red line @ 0
                        
            plt.xlabel("Time (s)")
            plt.ylabel("Force (N)")
            plt.title(("BL Ground Reaction Forces"))
            
            plt.savefig(self.save_path + "Episode " + str(i) + "/eval_plots/GRF/" + "08d BL Ground Reaction Forces.png")
            plt.clf()
       
    
    # Functions for data processing 
    def findones(self, a): 
        # functions from https://stackoverflow.com/questions/49634844/matplotlib-boolean-plot-rectangle-fill 
        isone = np.concatenate(([0], a, [0]))
        absdiff = np.abs(np.diff(isone))
        ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
        return np.clip(ranges, 0, len(a) - 1)


    def plotbooleans(self, ax, dictofbool):
        # functions from https://stackoverflow.com/questions/49634844/matplotlib-boolean-plot-rectangle-fill 
        ax.set_ylim([-1, len(dictofbool)])
        ax.set_yticks(np.arange(len(dictofbool.keys())))
        ax.set_yticklabels(dictofbool.keys())

        for i, (key, value) in enumerate(dictofbool.items()):
            indexes = self.findones(value)
            for idx in indexes:
                if idx[0] == idx[1]:
                    idx[1] = idx[1]+1
                ax.hlines(y=i, xmin=idx[0], xmax=idx[1], linewidth=15, color = "k")	
    
    
    def findContactNumber(self, data):
        
        contact = 0 # every leg starts in contact with the ground
        
        for i in range(len(data)- 4):    
            if data[i] == 1 and data[i+1] == 0  and data[i+4] == 1:
                for j in range(i, i+4):
                    data[j] = 1
        # Run another pass after intial pass to ensure all contacts are captured            
        for i in range(len(data)- 2):    
            if data[i] == 1 and data[i+1] == 0  and data[i+2] == 1:
                for j in range(i, i+2):
                    data[j] = 1
        
        for i in range(len(data)- 2):    
            if data[i] != data[i+1] and data[i] == 0:
                contact += 1
            #elif i == len(data) - 3:
            #    if data[i] == 1 and data[i - 5] == 0:
            #        contact += 1
                
        self.foot_contact_data.append(contact)    
             
        return contact, data  


    def writeSignificantInfotoFile(self):
        
        self.getJointRanges()

        for i in range(self.num_episode - 1):
            
            text_file_path = self.save_path + "Episode " + str(i) + "/eval_plots/" + "00 Eval Data" +".txt"
            with open(text_file_path, 'w'):
                pass

 
            with open(text_file_path, 'a') as file:
                
                file.write(self.lineBreak + '\n')
                file.write('OVERVIEW OF IMPORTANT INFORMATION' + '\n')
                file.write(self.lineBreak + '\n')
                file.write('Total Distance Traveled \t = {:.3f} m'.format(self.x_distance[i]) + '\n')
                file.write('Total Time Taken \t \t \t = {:.2f} s'.format(self.epiosde_data[i]["Time"].values[-1]) + '\n')
                file.write('Foot Contact Data \t \t \t FR: {} \t FL: {} \t BR: {} \t BL: {}'.format(self.foot_contact_data[i*4], self.foot_contact_data[i*4 + 1], self.foot_contact_data[i*4 + 2], self.foot_contact_data[i*4 + 3]) + '\n')
                file.write('Ad/Ab Hip Joint Ranges \n FR: {:.2f} , {:.2f} \n FL: {:.2f} , {:.2f} \n BR: {:.2f} , {:.2f} \n BL: {:.2f} , {:.2f}'.format(self.FR_adab_range[i*2], self.FR_adab_range[i*2 + 1], self.FL_adab_range[i*2], self.FL_adab_range[i*2 + 1], self.BR_adab_range[i*2], self.BR_adab_range[i*2 + 1], self.BL_adab_range[i*2], self.BL_adab_range[i*2 + 1]) + '\n')
                file.write('Hip Joint Ranges \n FR: {:.2f} , {:.2f} \n FL: {:.2f} , {:.2f} \n BR: {:.2f} , {:.2f} \n BL: {:.2f} , {:.2f}'.format(self.FR_hip_range[i*2], self.FR_hip_range[i*2 + 1], self.FL_hip_range[i*2], self.FL_hip_range[i*2 + 1], self.BR_hip_range[i*2], self.BR_hip_range[i*2 + 1], self.BL_hip_range[i*2], self.BL_hip_range[i*2 + 1]) + '\n')
                file.write('Knee Joint Ranges \n FR: {:.2f} , {:.2f} \n FL: {:.2f} , {:.2f} \n BR: {:.2f} , {:.2f} \n BL: {:.2f} , {:.2f}'.format(self.FR_knee_range[i*2], self.FR_knee_range[i*2 + 1], self.FL_knee_range[i*2], self.FL_knee_range[i*2 + 1], self.BR_knee_range[i*2], self.BR_knee_range[i*2 + 1], self.BL_knee_range[i*2], self.BL_knee_range[i*2 + 1]) + '\n')
                
                file.write(self.lineBreak + '\n')


    def changeRadtoDegrees(self, data):
        return data * 180 / np.pi


    def getJointRanges(self):
        
        for i in range(self.num_episode - 1):
            
            self.FR_adab_range.append((self.epiosde_data[i]["FR Hip Ad_Ab Angle"].max())*180 / np.pi)
            self.FR_adab_range.append((self.epiosde_data[i]["FR Hip Ad_Ab Angle"].min())*180 / np.pi)
            self.FR_hip_range.append((self.epiosde_data[i]["FR Hip Swing Angle"].max())*180 / np.pi)
            self.FR_hip_range.append((self.epiosde_data[i]["FR Hip Swing Angle"].min())*180 / np.pi)
            self.FR_knee_range.append((self.epiosde_data[i]["FR Knee Angle"].max())*180 / np.pi)
            self.FR_knee_range.append((self.epiosde_data[i]["FR Knee Angle"].min())*180 / np.pi)
            
            self.FL_adab_range.append((self.epiosde_data[i]["FL Hip Ad_Ab Angle"].max())*180 / np.pi)
            self.FL_adab_range.append((self.epiosde_data[i]["FL Hip Ad_Ab Angle"].min())*180 / np.pi)
            self.FL_hip_range.append((self.epiosde_data[i]["FL Hip Swing Angle"].max())*180 / np.pi)
            self.FL_hip_range.append((self.epiosde_data[i]["FL Hip Swing Angle"].min())*180 / np.pi)  
            self.FL_knee_range.append((self.epiosde_data[i]["FL Knee Angle"].max())*180 / np.pi)
            self.FL_knee_range.append((self.epiosde_data[i]["FL Knee Angle"].min())*180 / np.pi)
            
            self.BR_adab_range.append(self.epiosde_data[i]["BR Hip Ad_Ab Angle"].max()*180 / np.pi)
            self.BR_adab_range.append(self.epiosde_data[i]["BR Hip Ad_Ab Angle"].min()*180 / np.pi)
            self.BR_hip_range.append(self.epiosde_data[i]["BR Hip Swing Angle"].max()*180 / np.pi)
            self.BR_hip_range.append(self.epiosde_data[i]["BR Hip Swing Angle"].min()*180 / np.pi)
            self.BR_knee_range.append(self.epiosde_data[i]["BR Knee Angle"].max()*180 / np.pi)
            self.BR_knee_range.append(self.epiosde_data[i]["BR Knee Angle"].min()*180 / np.pi)
            
            self.BL_adab_range.append(self.epiosde_data[i]["BL Hip Ad_Ab Angle"].max()*180 / np.pi)
            self.BL_adab_range.append(self.epiosde_data[i]["BL Hip Ad_Ab Angle"].min()*180 / np.pi)
            self.BL_hip_range.append(self.epiosde_data[i]["BL Hip Swing Angle"].max()*180 / np.pi)
            self.BL_hip_range.append(self.epiosde_data[i]["BL Hip Swing Angle"].min()*180 / np.pi)
            self.BL_knee_range.append(self.epiosde_data[i]["BL Knee Angle"].max()*180 / np.pi)
            self.BL_knee_range.append(self.epiosde_data[i]["BL Knee Angle"].min()*180 / np.pi)


    def writeLegLoadingtoFile(self):
        
        self.splitContactData()
        self.getLegLoadingValues()
        
        for i in range(self.num_episode - 1):
            
            text_file_path = self.save_path + "Episode " + str(i) + "/eval_plots/GRF/" + "08e Leg Loading" +".txt"
            with open(text_file_path, 'w'):
                pass

 
            with open(text_file_path, 'a') as file:
                
                file.write(self.lineBreak + '\n')
                file.write('LEG LOADING INFORMATION' + '\n \n')
                file.write(self.lineBreak + '\n')
                file.write('FRONT RIGHT LEG'+ '\n')
                
                for i in range(self.num_episode_FR - 1):
                    
                    if self.FR_df[i]["FR Contact"].values[0] == 1:
                        file.write('Contact Phase' + '\n')
                        file.write('Time in Contact: {:.2f} s'.format((self.FR_df[i]["Time"].values[-1] - self.FR_df[i]["Time"].values[0])) + '\n')
                        file.write('Leg Loading: {:.3f} '.format(self.FR_loading[i]) + '\n \n')
                    else:
                        file.write('Swing Phase' + '\n')    
                        file.write('Time in Air: {:.2f} s'.format((self.FR_df[i]["Time"].values[-1] - self.FR_df[i]["Time"].values[0])) + '\n')
                        file.write('Leg Loading: {:.3f} '.format(self.FR_loading[i]) + '\n \n')
                
                file.write('Max Loading: {:.3f}'.format(self.FR_max_loading) + '\n') 
                file.write(self.lineBreak + '\n')       
                file.write('FRONT LEFT LEG'+ '\n')
                
                for j in range(self.num_episode_FL - 1):
                    
                    if self.FL_df[j]["FL Contact"].values[0] == 1:
                        file.write('Contact Phase' + '\n')
                        file.write('Time in Contact: {:.2f} s'.format((self.FL_df[j]["Time"].values[-1] - self.FL_df[j]["Time"].values[0])) + '\n')
                        file.write('Leg Loading: {:.3f}'.format(self.FL_loading[j]) + '\n \n')
                    else:
                        file.write('Swing Phase' + '\n')    
                        file.write('Time in Air: {:.2f} s'.format((self.FL_df[j]["Time"].values[-1] - self.FL_df[j]["Time"].values[0])) + '\n')
                        file.write('Leg Loading: {:.3f}'.format(self.FL_loading[j]) + '\n  \n')
                
                file.write('Max Loading: {:.3f}'.format(self.FL_max_loading) + '\n  \n') 
                file.write(self.lineBreak + '\n')        
                file.write('BACK RIGHT LEG'+ '\n')
                
                for k in range(self.num_episode_BR - 1):
                    
                    if self.BR_df[k]["BR Contact"].values[0] == 1:
                        file.write('Contact Phase' + '\n')
                        file.write('Time in Contact: {:.2f} s'.format((self.BR_df[k]["Time"].values[-1] - self.BR_df[k]["Time"].values[0])) + '\n')
                        file.write('Leg Loading: {:.3f}'.format(self.BR_loading[k]) + '\n \n')
                    else:
                        file.write('Swing Phase' + '\n')    
                        file.write('Time in Air: {:.2f} s'.format((self.BR_df[k]["Time"].values[-1] - self.BR_df[k]["Time"].values[0])) + '\n')
                        file.write('Leg Loading: {:.3f}'.format(self.BR_loading[k]) + '\n  \n')
                        
                file.write('Max Loading: {:.3f}'.format(self.BR_max_loading) + '\n \n') 
                file.write(self.lineBreak + '\n')       
                file.write('BACK LEFT LEG'+ '\n')

                for m in range(self.num_episode_BL - 1):
                
                    if self.BL_df[m]["BL Contact"].values[0] == 1:
                            file.write('Contact Phase' + '\n')
                            file.write('Time in Contact: {:.2f} s'.format((self.BL_df[m]["Time"].values[-1] - self.BL_df[m]["Time"].values[0])) + '\n')
                            file.write('Leg Loading: {:.3f}'.format(self.BL_loading[m]) + '\n \n')
                    else:
                            file.write('Swing Phase' + '\n')    
                            file.write('Time in Air: {:.2f} s'.format((self.BL_df[m]["Time"].values[-1] - self.BL_df[m]["Time"].values[0])) + '\n')
                            file.write('Leg Loading: {:.3f}'.format(self.BL_loading[m]) + '\n \n')
                
                file.write('Max Loading: {:.3f}'.format(self.BL_max_loading) + '\n')         
                file.write(self.lineBreak + '\n')


    def splitContactData(self): 
        
        self.new_df = self.full_data[["Time", "FR GRF", "FL GRF", "BR GRF", "BL GRF"]]
            
        self.new_df["FR Contact"] = np.concatenate(self.FR_smoothed_contacts)
        self.new_df["FL Contact"] = np.concatenate(self.FL_smoothed_contacts)
        self.new_df["BR Contact"] = np.concatenate(self.BR_smoothed_contacts)
        self.new_df["BL Contact"] = np.concatenate(self.BL_smoothed_contacts)
        
        self.FR_df = []
        self.FL_df = []
        self.BR_df = []
        self.BL_df = []
        
        indices1, prev1 = [], None
        indices2, prev2 = [], None
        indices3, prev3 = [], None
        indices4, prev4 = [], None

        # FIX THIS IS NOT SPLITTING DATA WELL 
        for index, row in self.new_df.iterrows():
            if prev1 is not None and row[5] != prev1 and index > 2: 
                indices1.append(index)
            prev1 = row[5]
            
        indices1.append(self.new_df.index[-1])
        self.num_episode_FR = len(indices1)
        
        for i in range(len(indices1) - 1):
            start = indices1[i]
            end = indices1[i+1]
            df_temp = self.new_df.iloc[start:end]
            self.FR_df.append(df_temp)
        
        for index, row in self.new_df.iterrows():
            if prev2 is not None and row[6] != prev2 and index > 2: 
                indices2.append(index)
            prev2 = row[6]
            
        indices2.append(self.new_df.index[-1])
        self.num_episode_FL = len(indices2)
        
        for i in range(len(indices2) - 1):
            start = indices2[i]
            end = indices2[i+1]
            df_temp = self.new_df.iloc[start:end]
            self.FL_df.append(df_temp)
            
        for index, row in self.new_df.iterrows():
            if prev3 is not None and row[7] != prev3 and index > 2: 
                indices3.append(index)
            prev3 = row[7]
            
        indices3.append(self.new_df.index[-1])
        self.num_episode_BR = len(indices3)
        
        for i in range(len(indices3) - 1):
            start = indices3[i]
            end = indices3[i+1]
            df_temp = self.new_df.iloc[start:end]
            self.BR_df.append(df_temp)
        
        for index, row in self.new_df.iterrows():
            if prev4 is not None and row[8] != prev4 and index > 2: 
                indices4.append(index)
            prev4 = row[8]
            
        indices4.append(self.new_df.index[-1])
        self.num_episode_BL = len(indices4)
        
        for i in range(len(indices4) - 1):
            start = indices4[i]
            end = indices4[i+1]
            df_temp = self.new_df.iloc[start:end]
            self.BL_df.append(df_temp)


    def getLegLoadingValues(self):
        
        self.FR_loading = []
        self.FL_loading = []
        self.BR_loading = []
        self.BL_loading = []
        
        self.FR_max_loading = 0 
        self.FL_max_loading = 0 
        self.BR_max_loading = 0
        self.BL_max_loading = 0
        
        for i in range(self.num_episode_FR - 1):
            y = self.FR_df[i]["FR GRF"].values
    
            if y.size == 0: 
                self.num_episode_BL += -1
                continue
            else: 
                area = simpson(y, dx=0.005)
                loading = area / (self.FR_df[i]["Time"].values[-1] - self.FR_df[i]["Time"].values[0])
                if math.isnan(loading): loading = 0 
                self.FR_loading.append(loading)
                if loading > self.FR_max_loading:
                    self.FR_max_loading = loading
                   
        i = 0 
        for i in range(self.num_episode_FL - 1):
            y = self.FL_df[i]["FL GRF"].values
            
            if y.size == 0:
                self.num_episode_BL += -1 
                continue
            else: 
                area = simpson(y, dx=0.005)
                loading = area / (self.FL_df[i]["Time"].values[-1] - self.FL_df[i]["Time"].values[0])
                if math.isnan(loading): loading = 0 
                self.FL_loading.append(loading)
                if loading > self.FL_max_loading:
                    self.FL_max_loading = loading
        
        i = 0     
        for i in range(self.num_episode_BR - 1):
            y = self.BR_df[i]["BR GRF"].values
            
            if y.size == 0: 
                self.num_episode_BL += -1
                continue
            else: 
                area = simpson(y, dx=0.005)
                loading = area / (self.BR_df[i]["Time"].values[-1] - self.BR_df[i]["Time"].values[0])
                if math.isnan(loading): loading = 0 
                self.BR_loading.append(loading)
                if loading > self.BR_max_loading:
                    self.BR_max_loading = loading
                
        
        i = 0      
        for i in range(self.num_episode_BL - 1):
            y = self.BL_df[i]["BL GRF"].values
            
            if y.size == 0:
                self.num_episode_BL += -1
            else: 
                area = simpson(y, dx=0.005)
                loading = area / (self.BL_df[i]["Time"].values[-1] - self.BL_df[i]["Time"].values[0])
                if math.isnan(loading): loading = 0 
                self.BL_loading.append(loading)
                if loading > self.BL_max_loading:
                    self.BL_max_loading = loading