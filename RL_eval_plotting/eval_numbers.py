import os
import csv
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.integrate import simpson
from scipy.optimize import curve_fit
from scipy.stats import kendalltau, spearmanr
from shapely.geometry import LineString
from scipy.optimize import differential_evolution
from scipy.signal import butter, filtfilt


class EvalModel_Ideal():
    
    def __init__(self, csv_file_path = any, save_path = any, time_to_ignore = 1):
        
        df = pd.read_csv(csv_file_path)
        self.full_data = df
        self.save_path = save_path
        # 102 corresponds to the first second of data which is normally when the robot is still accelerating
        self.num_points_to_ignore = time_to_ignore*100 + 2
        
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
            # make paths as neccessary
            os.makedirs((self.save_path + self.episode_save_path + "Eval Nums/Plots/"), exist_ok=True)
            os.makedirs((self.save_path + self.episode_save_path + "Eval Nums/Values/"), exist_ok=True)
            
    
        # Plot data for each episode
        for i in range(self.num_episode - 1):
            
            # Variables for storing R^2 number 
            self.spearman_joint_angles, self.cc_score_joint_angles = [], []
            self.spearman_pitch, self.cc_score_pitch = [], []
            self.spearman_foot_z_pos, self.cc_score_foot_z_pos = [], []
            self.spearman_foot_x_vs_z_pos, self.cc_score_foot_x_vs_z_pos = [], []

            # All the files to plot 
            self.determine_sinusoidal_JointAngles(i)
            self.determine_sinusoidal_BodyPitch(i)
            self.determine_sinusoidal_FootZPOS(i)
            self.determine_sinusoidal_FootxVSzPOS(i)
            self.determine_limitcycle_roll_vs_pitch(i)
            self.determine_limitcycle_angles(i)
        
            # self.write_to_ideal_csv(i)
            self.make_heatmaps(i)


    def determine_sinusoidal_JointAngles(self, i):
        # Plot angles with curve to see if resemble a curve at all 

        self.joint_headings = ["FR Hip Ad_Ab Angle", "FL Hip Ad_Ab Angle", "BR Hip Ad_Ab Angle", "BL Hip Ad_Ab Angle", 
                    "FR Hip Swing Angle", "FL Hip Swing Angle", "BR Hip Swing Angle", "BL Hip Swing Angle", 
                    "FR Knee Angle", "FL Knee Angle", "BR Knee Angle", "BL Knee Angle"]
        
        for j in range(12):
  
            x = self.epiosde_data[i]["Time"][self.num_points_to_ignore:]
            y1_1 = self.changeRadtoDegrees(self.epiosde_data[i][self.joint_headings[j]][self.num_points_to_ignore:])

            x_vals =  np.array(x.values)
            initial_guess = [np.ptp(y1_1)/2, 2*np.pi/(x_vals[-1]-x_vals[0]), 0, y1_1.mean()]
           
            try:
                # Fit the sinusoidal function to the data
                params, _ = curve_fit(sinusoidal, x, y1_1, p0=initial_guess, maxfev=5000)
                y_fit = sinusoidal(x, *params)

                # Calculate R-squared value
                spearman_corr, _ = spearmanr(y1_1, y_fit)
                self.spearman_joint_angles.append(spearman_corr)

                kendall_corr, _ = kendalltau(y1_1, y_fit)
                self.cc_score_joint_angles.append(kendall_corr)

                text = "R2 = {:.4f}, CC = {:.4f}".format(spearman_corr, kendall_corr)
                
                plt.close()
                plt.plot(x, y1_1, color = "C0", label=self.joint_headings[j])
                plt.plot(x, y_fit, label='Fitted sinusoidal function', color='red')
                plt.xlabel("Time (s)")
                plt.ylabel("Angle $(^\circ)$")
                plt.legend()
                plt.title((self.joint_headings[j]))
                
                x_position = np.max(x) * 1.3  # 110% of the max x value
                y_position = np.max(y1_1) * 1.1  # 95% of the max y value
                plt.text(x_position, y_position, text, horizontalalignment="right", verticalalignment="top")

                plt.savefig(self.save_path + "Episode " + str(i) + "/Eval Nums/Plots/0 " + self.joint_headings[j] + ".png", bbox_inches="tight")
                plt.clf()
                
            except RuntimeError: 
                
                self.spearman_joint_angles.append(math.nan) # NAN for no correlation to sinusoid
                self.cc_score_joint_angles.append(math.nan)

                plt.close()
                plt.plot(x, y1_1, color = "C0", label=self.joint_headings[j])
                plt.xlabel("Time (s)")
                plt.ylabel("Angle $(^\circ)$")
                plt.legend()
                plt.title(self.joint_headings[j])

                x_position = np.max(x) * 1.3  # 95% of the max x value
                y_position = np.max(y1_1) * 1.1  # 95% of the max y value
                plt.text(x_position, y_position, text, horizontalalignment="right", verticalalignment="top")

                plt.savefig(self.save_path + "Episode " + str(i) + "/Eval Nums/Plots/0 " + self.joint_headings[j] + ".png", bbox_inches="tight")
                plt.clf()
    
    def determine_sinusoidal_BodyPitch(self, i):
        # Plot pitch to see if similar to curve 

                
        x = self.epiosde_data[i]["Time"][self.num_points_to_ignore:]
        y1_1 = self.changeRadtoDegrees(self.epiosde_data[i]["Body Pitch"])[self.num_points_to_ignore:]

        # Initial guess for the parameters [Amplitude, Angular frequency, Phase shift, Offset]
        initial_guess = [np.max(y1_1), 40, 0, np.mean(y1_1)]
        
        # Fit the sinusoidal function to the data
        try:
            params, params_covariance = curve_fit(sinusoidal, x, y1_1, p0=initial_guess)

            # Generate y values using the fitted parameters
            y_fit = sinusoidal(x, *params)

            # Calculate R-squared value
            spearman_corr, _ = spearmanr(y1_1, y_fit)
            self.spearman_pitch.append(spearman_corr)

            kendall_corr, _ = kendalltau(y1_1, y_fit)
            self.cc_score_pitch.append(kendall_corr)

            text = "R2 = {:.4f}, CC = {:.4f}".format(spearman_corr, kendall_corr)
            
            plt.close()
            plt.plot(x, y1_1, color = "C0", label="Body Pitch")
            plt.plot(x, y_fit, label='Fitted sinusoidal function', color='red')
            plt.xlabel("Time (s)")
            plt.ylabel("Angle $(^\circ)$")
            plt.legend()
            plt.title(("Body Pitch"))
            
            x_position = np.max(x) * 1.3  # 110% of the max x value
            y_position = np.max(y1_1) * 1.1  # 95% of the max y value
            plt.text(x_position, y_position, text, horizontalalignment="right", verticalalignment="top")

            plt.savefig(self.save_path + "Episode " + str(i) + "/Eval Nums/Plots/1 Body Pitch" + ".png", bbox_inches="tight")
            plt.clf()
                
        except RuntimeError: 
            
            self.spearman_pitch.append(math.nan) # NAN for no correlation to sinusoid

            plt.close()
            plt.plot(x, y1_1, color = "C0", label="Body Pitch")
            plt.xlabel("Time (s)")
            plt.ylabel("Angle $(^\circ)$")
            plt.legend()
            plt.title("Body Pitch")

            x_position = np.max(x) * 1.3  # 95% of the max x value
            y_position = np.max(y1_1) * 1.1  # 95% of the max y value
            plt.text(x_position, y_position, text, horizontalalignment="right", verticalalignment="top")

            plt.savefig(self.save_path + "Episode " + str(i) + "/Eval Nums/Plots/1 Body Pitch" + ".png", bbox_inches="tight")
            plt.clf()
        
    def determine_sinusoidal_FootZPOS(self, i):
        # Plot pitch to see if similar to curve __,^,__,^,__

        headings = ["FR Foot Position Z", "FL Foot Position Z", "BR Foot Position Z", "BL Foot Position Z"]
        colours = ["C0", "C1", "C2", "C3"]
        
        for j in range(4):
        
            x = self.epiosde_data[i]["Time"][self.num_points_to_ignore:]
            y1_1 = self.epiosde_data[i][headings[j]][self.num_points_to_ignore:]
            y1_1 = y1_1 - 0.02

            # Initial guess for the parameters [Amplitude, Angular frequency, Phase shift, Offset]
            initial_guess = [np.max(y1_1), 40, 0]
            
            # Fit the sinusoidal function to the data
            try:
                params, params_covariance = curve_fit(piecewise_sine, x, y1_1, p0=initial_guess)

                # Generate y values using the fitted parameters
                y_fit = piecewise_sine(x, *params)

                # Calculate R-squared value
                spearman_corr, _ = spearmanr(y1_1, y_fit)
                self.spearman_foot_z_pos.append(spearman_corr)

                kendall_corr, _ = kendalltau(y1_1, y_fit)
                self.cc_score_foot_z_pos.append(kendall_corr)

                text = "R2 = {:.4f}, CC = {:.4f}".format(spearman_corr, kendall_corr)
                
                plt.close()
                plt.plot(x, y1_1, color = colours[j], label=headings[j])
                plt.plot(x, y_fit, label='Fitted sinusoidal function', color='red')
                plt.xlabel("Time (s)")
                plt.ylabel("Angle $(^\circ)$")
                plt.legend()
                plt.title((headings[j]))
                
                x_position = np.max(x) * 1.3  # 110% of the max x value
                y_position = np.max(y1_1) * 1.1  # 95% of the max y value
                plt.text(x_position, y_position, text, horizontalalignment="right", verticalalignment="top")

                plt.savefig(self.save_path + "Episode " + str(i) + "/Eval Nums/Plots/2 " + headings[j] + ".png", bbox_inches="tight")
                plt.clf()
                
            except RuntimeError: 
                
                self.spearman_foot_z_pos.append(math.nan) # NAN for no correlation to sinusoid

                plt.close()
                plt.plot(x, y1_1, color = colours[j], label=headings[j])
                plt.xlabel("Time (s)")
                plt.ylabel("Angle $(^\circ)$")
                plt.title(headings[j])

                x_position = np.max(x) * 1.3  # 95% of the max x value
                y_position = np.max(y1_1) * 1.1  # 95% of the max y value
                plt.text(x_position, y_position, text, horizontalalignment="right", verticalalignment="top")

                plt.savefig(self.save_path + "Episode " + str(i) + "/Eval Nums/Plots/2 " + headings[j] + ".png", bbox_inches="tight")
                plt.clf()
    
    def determine_sinusoidal_FootxVSzPOS(self, i):
        # Plot pitch to see if similar to curve ,^,,^,
        # can produce strange results because x vlaues sometimes repeat 

        z_pos = ["FR Foot Position Z", "FL Foot Position Z", "BR Foot Position Z", "BL Foot Position Z"]
        x_pos = ["FR Foot Position X", "FL Foot Position X", "BR Foot Position X", "BL Foot Position X"]
        plot_headings = ["FR Foot Position X vs Z", "FL Foot Position X vs Z", "BR Foot Position X vs Z", "BL Foot Position X vs Z"]
        colours = ["C0", "C1", "C2", "C3"]
        
        for j in range(4):
  
            x = self.epiosde_data[i][x_pos[j]] 
            y1_1 = self.epiosde_data[i][z_pos[j]] - 0.02
            

            # Initial guess for the parameters [Amplitude, Angular frequency, Phase shift, Offset]
            initial_guess = [np.max(y1_1), 40, 0]
            
            # Fit the sinusoidal function to the data
            try:
                params, params_covariance = curve_fit(positive_sine, x, y1_1, p0=initial_guess)

                # Generate y values using the fitted parameters
                y_fit = positive_sine(x, *params)

                # Calculate R-squared value
                spearman_corr, _ = spearmanr(y1_1, y_fit)
                self.spearman_foot_x_vs_z_pos.append(spearman_corr)

                kendall_corr, _ = kendalltau(y1_1, y_fit)
                self.cc_score_foot_x_vs_z_pos.append(kendall_corr)

                text = "R2 = {:.4f}, CC = {:.4f}".format(spearman_corr, kendall_corr)
                
                plt.close()
                plt.plot(x, y1_1, color = colours[j])
                plt.plot(x, y_fit, label='Fitted sinusoidal function', color='red')
                plt.xlabel("X Pos")
                plt.ylabel("Z Pos")
                plt.legend()
                plt.title((plot_headings[j]))
                
                x_position = np.max(x) * 1.3  # 110% of the max x value
                y_position = np.max(y1_1) * 1.1  # 95% of the max y value
                plt.text(x_position, y_position, text, horizontalalignment="right", verticalalignment="top")

                plt.savefig(self.save_path + "Episode " + str(i) + "/Eval Nums/Plots/3 " + plot_headings[j] + ".png", bbox_inches="tight")
                plt.clf()
                
            except RuntimeError: 
                
                self.spearman_foot_x_vs_z_pos.append(math.nan) # NAN for no correlation to sinusoid

                plt.close()
                plt.plot(x, y1_1, color = colours[j])
                plt.xlabel("X Pos")
                plt.ylabel("Z Pos")
                plt.title(plot_headings[j])

                x_position = np.max(x) * 1.3  # 95% of the max x value
                y_position = np.max(y1_1) * 1.1  # 95% of the max y value
                plt.text(x_position, y_position, text, horizontalalignment="right", verticalalignment="top")

                plt.savefig(self.save_path + "Episode " + str(i) + "/Eval Nums/Plots/3 " + plot_headings[j] + ".png", bbox_inches="tight")
                plt.clf()

    def determine_limitcycle_roll_vs_pitch(self, i):
        # Plot roll vs pitch to see if forms limit cycle 
        # ignore first 1 second of data = first second is 102 data points and is normally where the robot is still accelerating

        x = self.changeRadtoDegrees(self.epiosde_data[i]["Body Pitch"])[self.num_points_to_ignore:]
        y = self.changeRadtoDegrees(self.epiosde_data[i]["Body Roll"])[self.num_points_to_ignore:]
        mid_point = 0 
        self.std_array_roll = {}
        self.std_array_pitch = {}
        # Plot the line
        plt.close()
        plt.plot(x, y)
        plt.title(("Body Pitch vs Roll"))
        plt.xlabel("Pitch $(^\circ)$")
        plt.ylabel("Roll $(^\circ)$")
        plt.grid()

        # Detect self-intersections
        # num_intersections, intersection_points, max_streak = self.detect_self_intersections(x.values, y.values, tolerance=1e-2)

        # # Plot intersection points
        # if intersection_points:
        #     intersection_points = np.array(intersection_points)
        #     plt.scatter(intersection_points[:, 0], intersection_points[:, 1], color='red', zorder=5, label='Intersections')
        #     plt.legend()

        # # Print the number of intersections and max streak
        # self.pitch_roll_intersections = num_intersections
        # self.pitch_roll_max_streak = max_streak
        # self.num_steps = len(x)
        # print(f'Number of self-intersections: {num_intersections}')
        # print(f'Number of self-intersections as % of total points: {num_intersections/len(x)}')
        # print(f'Maximum consecutive intersections: {max_streak}')
        # print(f'Number of consecutive intersections as % of total points: {max_streak/len(x)}')

        # Detect limit-cycle-ness 
        x_zero_point_array, y_zero_point_array, x_zero_point_std_top, x_zero_point_std_bottom, y_zero_point_std_left,  y_zero_point_std_right = self.detectIntersectionswithZERO_points(x.values, y.values, mid_point)
        x_zero_point_array = np.array(x_zero_point_array)
        y_zero_point_array = np.array(y_zero_point_array)
        self.std_array_roll[ "Roll mid x-point (top, bottom)"] = [x_zero_point_std_top, x_zero_point_std_bottom]
        self.std_array_pitch["Pitch zero velocity point (left, right)"] = [y_zero_point_std_left, y_zero_point_std_right]
        self.std_array = [[(x_zero_point_std_top +  x_zero_point_std_bottom)/2],[(y_zero_point_std_left + y_zero_point_std_right)/2]]

        plt.scatter([mid_point]*len(x_zero_point_array), x_zero_point_array, color='red', zorder=5, label='Intersections')
        plt.scatter(y_zero_point_array, [0]*len(y_zero_point_array), color='blue', zorder=5, label='Intersections')
        plt.legend()

        plt.savefig(self.save_path + "Episode " + str(i) + "/Eval Nums/Plots/4 Pitch vs Roll" + ".png", bbox_inches="tight")
        plt.clf()
    
    def determine_limitcycle_angles(self, i):
        # ignore first 1 second of data = first second is 102 data points and is normally where the robot is still accelerating

        x_data = ["Body Roll", "Body Pitch", "Body Yaw", 
                    "FR Hip Ad_Ab Angle", "FR Hip Swing Angle", "FR Knee Angle",
                    "FL Hip Ad_Ab Angle", "FL Hip Swing Angle", "FL Knee Angle",
                    "BR Hip Ad_Ab Angle", "BR Hip Swing Angle", "BR Knee Angle",
                    "BL Hip Ad_Ab Angle", "BL Hip Swing Angle", "BL Knee Angle"]
        y_data = ["Body Ang Vel x", "Body Ang Vel y", "Body Ang Vel z",
                    "FR Hip Angle Vel", "FR Knee Angle Vel", "FR Ankle Angle Vel",
                    "FL Hip Angle Vel", "FL Knee Angle Vel", "FL Ankle Angle Vel",
                    "BR Hip Angle Vel", "BR Knee Angle Vel", "BR Ankle Angle Vel",
                    "BL Hip Angle Vel", "BL Knee Angle Vel", "BL Ankle Angle Vel"]
        headings = ["5a Body Roll", "5a Body Pitch", "5a Body Yaw", 
                "5b FR Hip Ad_Ab", "5b FR Hip Swing", "5b FR Knee",
                "5c FL Hip Ad_Ab", "5c FL Hip Swing", "5c FL Knee",
                "5d BR Hip Ad_Ab", "5d BR Hip Swing", "5d BR Knee",
                "5e BL Hip Ad_Ab", "5e BL Hip Swing", "5e BL Knee"]
        colours = ["C4", "C4", "C4", "C0", "C0", "C0", "C1", "C1", "C1", "C2", "C2", "C2", "C3", "C3", "C3"]

        self.std_array_zero_velocity = {}
        self.std_array_zero_degrees = {}
        self.std_array_joint_angles = None
        
        for j in range(len(x_data)):
            x = self.changeRadtoDegrees(self.epiosde_data[i][x_data[j]])[self.num_points_to_ignore:]
            y = self.changeRadtoDegrees(self.epiosde_data[i][y_data[j]])[self.num_points_to_ignore:]

            # Plot the line
            plt.close()
            plt.plot(x, y, color = colours[j])
            plt.title((headings[j]))
            plt.xlabel(r"$\theta  \: (^\circ)$")
            plt.ylabel(r"$\dot{\theta}  \: (^\circ / \,s)$")
            plt.grid()
            
            mid_point = x.mean()

            # Detect limit-cycle-ness 
            x_zero_point_array, y_zero_point_array, x_zero_point_std_top, x_zero_point_std_bottom, y_zero_point_std_left,  y_zero_point_std_right = self.detectIntersectionswithZERO_points(x.values, y.values, mid_point)
            x_zero_point_array = np.array(x_zero_point_array)
            y_zero_point_array = np.array(y_zero_point_array)
            self.std_array_zero_degrees[x_data[j] + " mid x-point"] = [x_zero_point_std_top, x_zero_point_std_bottom]
            self.std_array_zero_velocity[x_data[j] + " zero velocity point"] = [y_zero_point_std_left, y_zero_point_std_right]
            data_to_concatenate = [[x_zero_point_std_top, x_zero_point_std_bottom],[y_zero_point_std_left, y_zero_point_std_right]]
            
            if self.std_array_joint_angles is None:
                self.std_array_joint_angles = data_to_concatenate
            else:
                self.std_array_joint_angles = np.concatenate((self.std_array_joint_angles, data_to_concatenate), axis=0)

            # print(f"Standard deviation of zero crossing points for {headings[j]}: {x_zero_point_std}, {y_zero_point_std}")

            plt.scatter([mid_point]*len(x_zero_point_array), x_zero_point_array, color='red', zorder=5, label='Intersections')
            plt.scatter(y_zero_point_array, [0]*len(y_zero_point_array), color='blue', zorder=5, label='Intersections')
            plt.legend()
            
            plt.savefig(self.save_path + "Episode " + str(i) + "/Eval Nums/Plots/" + headings[j] + ".png", bbox_inches="tight")
            plt.clf()
        
        avg_std_array = np.mean(self.std_array_joint_angles, axis=1)
        std_array_temp1_x = []
        std_array_temp2_y = []

        for i in range(len(avg_std_array)):
            if i % 2 == 0:
                std_array_temp1_x.append(avg_std_array[i])
            else:
                std_array_temp2_y.append(avg_std_array[i])
   
        self.std_body_array = [std_array_temp1_x[:3], std_array_temp2_y[:3]] # Limit cycle averag std for body angles
        self.std_array_joint_angles_hip = [std_array_temp1_x[3:7], std_array_temp2_y[3:7]] # Limit cycle average std for hip angles
        self.std_array_joint_angles_thigh = [std_array_temp1_x[7:11], std_array_temp2_y[7:11]] # Limit cycle average std for thigh angles
        self.std_array_joint_angles_calf = [std_array_temp1_x[11:15], std_array_temp2_y[11:15]] # Limit cycle average std for calf angles
    

    def write_to_ideal_csv(self, i):
        # Write all the data to a csv file
        with open(self.save_path + "Episode " + str(i) + "/Eval Nums/Values/ideal_values.csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Joint Angle", "R2", "Kendall's Tau"])

            for j in range(12):
                writer.writerow(["Joint Angle " + self.joint_headings[j], self.spearman_joint_angles[j], self.cc_score_joint_angles[j]])
            
            writer.writerow(["Body Pitch", self.spearman_pitch, self.cc_score_pitch])
            writer.writerow(["Foot Z Pos", self.spearman_foot_z_pos, self.cc_score_foot_z_pos])
            writer.writerow(["Foot X vs Z Pos", self.spearman_foot_x_vs_z_pos, self.cc_score_foot_x_vs_z_pos])
            writer.writerow(["Pitch vs Roll, intersections, max streak, percent intersec, percent max",
                             self.pitch_roll_intersections, self.pitch_roll_max_streak, self.pitch_roll_intersections/self.num_steps, self.pitch_roll_max_streak/self.num_steps])
            writer.writerow(["Zero Crossing Points std", self.std_array_zero_degrees, self.std_array_zero_velocity]) # for all angles
                 
    def make_heatmaps(self, i):
        
        fig = plt.figure(figsize=(22, 9))
        gs = fig.add_gridspec(13, 26, wspace=2, hspace=2)
        
        ax1 = fig.add_subplot(gs[0:13, 0:10])
        ax4 = fig.add_subplot(gs[0:6, 11:20])
        ax5 = fig.add_subplot(gs[7:13, 11:20])
        ax2 = fig.add_subplot(gs[0:3, 21:26])
        ax3 = fig.add_subplot(gs[4:7, 21:26])
        
        
        
        headings_top = ["FR", "FL", "BR", "BL"]
        headings_side = ["Hip Angle - " + r"$\rho$", "Hip Angle - " + r"$\tau$", "Thigh Angle - " + r"$\rho$", "Thigh Angle - " + r"$\tau$", 
                         "Calf Angle - " + r"$\rho$", "Calf Angle - " + r"$\tau$", "Foot heights - " + r"$\rho$", "Foot heights - " + r"$\tau$",
                         "Foot Z vs X - " + r"$\rho$", "Foot Z vs X - " + r"$\tau$"]

        data_angles = np.zeros((6, 4))
        # data processing for heatmap 
        for j in range(3):
            for k in range(4):
                data_angles[j*2][k] = self.spearman_joint_angles[k + j*4]
                data_angles[j*2+1][k] = self.cc_score_joint_angles[k + j*4]
    
        reshaped_s_foot_z = np.reshape(self.spearman_foot_z_pos, (1, -1))
        reshaped_t_foot_z = np.reshape(self.cc_score_foot_z_pos, (1, -1))
        reshaped_s_foot_x_z = np.reshape(self.spearman_foot_x_vs_z_pos, (1, -1))
        reshaped_t_foot_x_z = np.reshape(self.cc_score_foot_x_vs_z_pos, (1, -1))
        
        data = np.concatenate((data_angles, reshaped_s_foot_z, reshaped_t_foot_z, 
                       reshaped_s_foot_x_z, reshaped_t_foot_x_z), axis=0)
        
        # Making heatmap for leg angles and foot positions
        im1, cbar = heatmap(data, headings_side, headings_top, ax=ax1,
                    cmap="twilight_shifted", vmin=-1, vmax=1, cbarlabel="Correlation Coefficients")
        texts = annotate_heatmap(im1, data, valfmt="{x:.2f}", size=10, textcolors=("black", "black"))

        # fig.suptitle("Correlation Coefficients of 'Ideal' behaviour", fontsize=16)
        # plt.savefig(self.save_path + "Episode " + str(i) + "/Eval Nums/Plots/8a Ideal Heatmap.png", bbox_inches="tight")
        # plt.clf()
        
        # Making heatmaps for pitch
        headings_top_1 = ["Spearmans", "Kendall's"]
        headings_side_1 = ["Body Pitch"]
        headings_top_2 = ["Pitch", "Roll", "Yaw", "Pitch vs Roll"]
        headings_side_2 = ["X std", "Y std"]
        
        data_pitch_1 = np.array([self.spearman_pitch, self.cc_score_pitch]).reshape((1, 2))
        data_pitch_2 = np.concatenate((self.std_body_array, self.std_array), axis=1).reshape((2, 4))
        
        cbar_kw = {"shrink": .9}
        im1, cbar = heatmap(data_pitch_1, headings_side_1, headings_top_1, ax=ax2,
                    cmap="Purples", vmin=-1, vmax=1, cbarlabel="Correlation Coefficients", cbar_kw=cbar_kw)
        texts = annotate_heatmap(im1, data_pitch_1, valfmt="{x:.2f}", size=10, textcolors=("black", "black"), threshold=0)
        
        im2, cbar = heatmap(data_pitch_2, headings_side_2, headings_top_2, ax=ax3,
                    cmap="twilight_shifted", vmin=-30, vmax=30, cbarlabel="STD of Zero Crossing Points", cbar_kw=cbar_kw)
        texts = annotate_heatmap(im2, data_pitch_2, valfmt="{x:.2f}", size=10, textcolors=("black", "white"))

        # fig.suptitle("Pitch and Roll Analysis", fontsize=16)
        # plt.savefig(self.save_path + "Episode " + str(i) + "/Eval Nums/Plots/8b Pitch Heatmap.png", bbox_inches="tight")
        # plt.clf()
        
        # Angle limit cycle plot
        headings_top_1 = ["FR", "FL", "BR", "BL"]
        headings_side_1 = ["Hip X std", "Thigh X std", "Calf X std"]
        headings_side_2 = ["Hip Y std", "Thigh Y std", "Calf Y std"]
        
        data_angles_1 = np.concatenate((self.std_array_joint_angles_hip[0], self.std_array_joint_angles_thigh[0], self.std_array_joint_angles_calf[0]), axis=0).reshape((3, 4))
        data_angles_2 = np.concatenate((self.std_array_joint_angles_hip[1], self.std_array_joint_angles_thigh[1], self.std_array_joint_angles_calf[1]), axis=0).reshape((3, 4)) 
        
        cbar_kw = {"shrink": .9}
        im1, cbar = heatmap(data_angles_1, headings_side_1, headings_top_1, ax=ax4,
                    cmap="Blues", vmin=10, vmax=400, cbarlabel="X STD of Zero Crossing Points", cbar_kw=cbar_kw)
        texts = annotate_heatmap(im1, data_angles_1, valfmt="{x:.1f}", size=10, textcolors=("black", "black"), threshold=0)
        
        im2, cbar = heatmap(data_angles_2, headings_side_2, headings_top_1, ax=ax5,
                    cmap="Oranges", vmin=-5.5, vmax=5.5, cbarlabel="Y STD of Zero Crossing Points", cbar_kw=cbar_kw)
        texts = annotate_heatmap(im2, data_angles_2, valfmt="{x:.3f}", size=10, textcolors=("black", "black"))

        fig.suptitle("'Ideal' Analysis", fontsize=24)
        plt.savefig(self.save_path + "Episode " + str(i) + "/Eval Nums/Plots/8b Ideal Analysis Heatmap.png", bbox_inches="tight")
        plt.clf()
        # fig.suptitle("Joint Angles Analysis", fontsize=16)
        # plt.savefig(self.save_path + "Episode " + str(i) + "/Eval Nums/Plots/8b Joint Angles Heatmap.png", bbox_inches="tight")
        # plt.clf()
        
    # Additional neccessary functions 
    def changeRadtoDegrees(self, data):
        return data * 180 / np.pi
    
    # Determine how many interscetions have happened
    def detect_self_intersections(self, x, y, tolerance=1e-5):
        # Create a LineString from the points
        line = LineString(np.column_stack((x, y)))

        # Find self-intersections
        intersection_points = []
        num_intersections = 0
        current_streak = 0
        max_streak = 0

        for i in range(len(x) - 1):
            segment = LineString([(x[i], y[i]), (x[i+1], y[i+1])])
            found_intersection = False
            for j in range(i + 2, len(x) - 1):
                other_segment = LineString([(x[j], y[j]), (x[j+1], y[j+1])])
                if segment.distance(other_segment) < tolerance:
                    intersection_point = segment.intersection(other_segment)
                    if intersection_point.is_empty:
                        continue
                    elif intersection_point.geom_type == 'Point':
                        intersection_points.append((intersection_point.x, intersection_point.y))
                        num_intersections += 1
                        found_intersection = True
                    elif intersection_point.geom_type == 'MultiPoint':
                        for point in intersection_point:
                            intersection_points.append((point.x, point.y))
                            num_intersections += 1
                        found_intersection = True

            if found_intersection:
                current_streak += 1
            else:
                if current_streak > max_streak:
                    max_streak = current_streak
                current_streak = 0

        if current_streak > max_streak:
            max_streak = current_streak

        return num_intersections, intersection_points, max_streak
    
    def detectIntersectionswithZERO_points(self, x, y, x_mid_point):
        y_zero_point_l = []
        y_zero_point_r = []
        x_zero_point_top = []
        x_zero_point_bottom = []

        for i in range(len(x) - 1):
            # Detect zero crossing in y
            if y[i] * y[i + 1] < 0:  # Crossing zero
                # Linear interpolation to find the zero-crossing point
                x_zero = x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
                if x_zero > x_mid_point:
                    y_zero_point_r.append(x_zero)
                else: y_zero_point_l.append(x_zero)
            elif y[i] == 0:  # Directly at zero
                if x[i] > x_mid_point:
                    y_zero_point_r.append(x[i])
                else: y_zero_point_l.append(x[i])
            
            # Detect zero crossing in x
            if x_mid_point == 0:
                if x[i] * x[i + 1] < 0:  # Crossing zero
                    # Linear interpolation to find the zero-crossing point
                    y_zero = y[i] - x[i] * (y[i + 1] - y[i]) / (x[i + 1] - x[i])
                    if y_zero > 0:
                        x_zero_point_top.append(y_zero)
                    else: x_zero_point_bottom.append(y_zero)
                elif x[i] == x_mid_point:  # Directly at the midpoint
                    if y[i] > 0:
                        x_zero_point_top.append(y[i])
                    else: x_zero_point_bottom.append(y[i])
            else: 
                if (x[i] - x_mid_point) * (x[i + 1] - x_mid_point) < 0:  # Crossing the midpoint
                    # Linear interpolation to find the crossing point
                    y_crossing = y[i] - (x[i] - x_mid_point) * (y[i + 1] - y[i]) / (x[i + 1] - x[i])
                    if y_crossing > 0:
                        x_zero_point_top.append(y_crossing)
                    else:
                        x_zero_point_bottom.append(y_crossing)
                elif x[i] == x_mid_point:  # Directly at the midpoint
                    if y[i] > 0:
                        x_zero_point_top.append(y[i])
                    else:
                        x_zero_point_bottom.append(y[i])

        y_zero_point_std_left = np.std(y_zero_point_l)
        y_zero_point_std_right = np.std(y_zero_point_r)
        x_zero_point_std_top = np.std(x_zero_point_top)
        x_zero_point_std_bottom = np.std(x_zero_point_bottom)

        x_zero_point = np.concatenate((x_zero_point_top, x_zero_point_bottom))
        y_zero_point = np.concatenate((y_zero_point_l, y_zero_point_r))
        
        return x_zero_point, y_zero_point, x_zero_point_std_top, x_zero_point_std_bottom, y_zero_point_std_left,  y_zero_point_std_right
    
class EvalModel_Skewness():
    
    def __init__(self, csv_file_path = any, save_path = any, time_to_ignore = 1):
        
        df = pd.read_csv(csv_file_path)
        self.full_data = df
        self.save_path = save_path
        # 102 corresponds to the first second of data which is normally when the robot is still accelerating
        self.num_points_to_ignore = time_to_ignore*100 + 2
        
        
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
            # make paths as neccessary
            os.makedirs((self.save_path + self.episode_save_path + "Eval Nums/Plots/"), exist_ok=True)
            os.makedirs((self.save_path + self.episode_save_path + "Eval Nums/Values/"), exist_ok=True)

        # Plot data for each episode
        for i in range(self.num_episode - 1):
            
            self.total_time = self.epiosde_data[i]["Time"].iloc[-1]
            
            self.foot_contact_data = []
            
            # All the files to execute
            self.roll_body_deviations(i)
            self.yaw_body_deviations(i)
            self.foot_deviations(i)
            self.num_contacts(i)
            self.write_to_skewness_csv(i)
            self.make_heatmaps(i)
    
    def roll_body_deviations(self,i):
        # Get roll devation

        data = self.changeRadtoDegrees(self.epiosde_data[i]["Body Roll"])
        self.roll_max_value = data.max()
        self.roll_min_value = data.min()
        self.roll_average_value = data.mean()
        self.roll = [self.roll_max_value, self.roll_min_value, self.roll_average_value]
            
    def yaw_body_deviations(self,i):
        # Get yaw devation

        data = self.changeRadtoDegrees(self.epiosde_data[i]["Body Yaw"])
        self.yaw_max_value = data.max()
        self.yaw_min_value = data.min() 
        self.yaw_average_value = data.mean()
        self.yaw = [self.yaw_max_value, self.yaw_min_value, self.yaw_average_value]

    def foot_deviations(self, i):

        headings = ["FR Foot Position Y", "FL Foot Position Y", "BR Foot Position Y", "BL Foot Position Y"]
        self.foot_deviation = {}
        self.foot_deviation_array = []
        self.foot_deviation_array = np.empty((4, 3))

        for j in range(4):
            data = self.epiosde_data[i][headings[j]]
            self.foot_deviation[headings[j] + " max deviation"] = data.max()
            self.foot_deviation[headings[j] + " min deviation"] = data.min()
            self.foot_deviation[headings[j] + " average deviation"] = data.mean()
            self.foot_deviation_array[j] = [data.max(), data.min(), data.mean()]
    
    def num_contacts(self, i):
        #determine length of contact for each foot

        x_data = ["FR Contact", "FL Contact", "BR Contact", "BL Contact"]
        self.contact_headings = x_data
        self.contact_num = {}
        self.contact_lengths = {}
        self.contact_as_percentage = {} 
        total_contacts = []   

        for j in range(len(x_data)):

            data = self.epiosde_data[i][x_data[j]]
            num_contacts, contact_lengths = count_contacts(data)
            total_contacts.append(num_contacts)
            self.contact_num[x_data[j] + " num contacts"] = num_contacts
            self.contact_lengths[x_data[j] + " contact lengths max and min and num min"] = [np.max(contact_lengths), np.min(contact_lengths), 
                                                                                            np.count_nonzero(contact_lengths == np.min(contact_lengths))]
        
        sum_contacts = sum(total_contacts) 

        for j in range(len(x_data)):
            self.contact_as_percentage[x_data[j] + " contact as percentage"] = (total_contacts[j] / sum_contacts) * 100

        # print(self.contact_num) # number of contacts straight 
        # print(self.contact_lengths) # max length of contact, min length of contact, then number of times the contact is a minimum
        # print(self.contact_as_percentage) # percentage between 0 and 1 of contact number as total of contacts 
    
    def write_to_skewness_csv(self, i):

        with open(self.save_path + "Episode " + str(i) + "/Eval Nums/Values/skewness_values.csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Roll Max Value deg", "Roll Average Value deg", self.roll_max_value, self.roll_average_value])
            writer.writerow(["Yaw Max Value deg", "Yaw Average Value deg", self.yaw_max_value, self.yaw_average_value])
            writer.writerow(["Foot Deviation (m)", self.foot_deviation]) # has max and mean 
            writer.writerow(["Number of Contacts", self.contact_num])
    
    def make_heatmaps(self, i):

        headings_y = ["FR", "FL", "BR", "BL"]
        headings_x1 = ["Max", "Min", "Num Min"] 
        headings_y2 = ["Num", "% Of Total"]

        # data processing for heatmap 
        data1 = np.array([list(self.contact_lengths["FR Contact contact lengths max and min and num min"]), 
                        list(self.contact_lengths["FL Contact contact lengths max and min and num min"]), 
                        list(self.contact_lengths["BR Contact contact lengths max and min and num min"]), 
                        list(self.contact_lengths["BL Contact contact lengths max and min and num min"])])
        data2 = np.array([list(self.contact_num.values()), list(self.contact_as_percentage.values())])
        
        fig3 = plt.figure(figsize=(15, 10))
        gs = fig3.add_gridspec(14, 5, wspace=0.4)  # Reduced number of columns and added wspace
        f3_ax1 = fig3.add_subplot(gs[0:14, 0:2])  # Skewness Values (left)
        f3_ax2 = fig3.add_subplot(gs[0:9, 2:5])   # Contact Lengths (top right)
        f3_ax3 = fig3.add_subplot(gs[10:14, 2:5]) # Number of contacts (bottom right)

        fig3.suptitle('Contact and Skewness Analysis', fontsize=16, y=0.95)
        
        #fig, ax = plt.subplots(2, 1, figsize=(5, 8))
        cbar_kw = {"shrink": .9}
        im1, cbar = heatmap_columns(data1, headings_y, headings_x1, ax=f3_ax2, color_columns=2,
                   cmap="Blues", vmin=0, vmax=30, cbarlabel="Contact Lengths", cbar_kw=cbar_kw)
        texts = annotate_heatmap_col(im1, data1, valfmt="{x:.2f}", size=12, textcolors=("black", "black"))

        cbar_kw = {"shrink": .9}
        im2, cbar = heatmap_columns(data2, headings_y2, headings_y, ax=f3_ax3, color_rows=1,
                   cmap="Oranges", vmin=0, vmax=100, cbarlabel="Contact Numbers", cbar_kw=cbar_kw)
         
        texts = annotate_heatmap_col(im2, data2, valfmt="{x:.1f}", size=12, textcolors=("black", "black"))
        #fig.tight_layout(rect=[0, 0, 1, 0.95])
        #fig.suptitle("Contact Lengths and Numbers", fontsize=16)
        # plt.savefig(self.save_path + "Episode " + str(i) + "/Eval Nums/Plots/7a Contact Heatmap.png", bbox_inches="tight")
        # plt.clf()
        
        # Heatmap for general skewness values (max, min, avg)
        data = np.concatenate((np.array([self.roll]), np.array([self.yaw]), self.foot_deviation_array))
        row_headings = ["Body Roll Angle", "Body Yaw Angle", "FR Foot Y-position", "FL Foot Y-position", "BR Foot Y-position", "BL Foot Y-position"]
        column_headings = ["Max", "Min", "Average"]
        
        #fig, ax = plt.subplots(1, 1, figsize=(5, 8))
        cbar_kw = {"shrink": .85}
        im1, cbar = heatmap_columns(data, row_headings, column_headings, ax=f3_ax1, color_columns=2,
                   cmap="Purples", vmin=-2, vmax=2, cbarlabel="Skewness Values", cbar_kw=cbar_kw)
        texts = annotate_heatmap_col(im1, data, valfmt="{x:.2f}", size=12, textcolors=("black", "black"))
        
        #fig.suptitle("Skewness Values", fontsize=16)
        plt.savefig(self.save_path + "Episode " + str(i) + "/Eval Nums/Plots/7b Skewness Heatmap.png", bbox_inches="tight")
        plt.clf()
        

    # Additional neccessary functions
    def changeRadtoDegrees(self, data):
        return data * 180 / np.pi

    def count_contacts(self, data):
        num_contacts = 0
        prev_contact = 0

        for contact in data:
            if contact == 1 and prev_contact == 0:
                num_contacts += 1
            prev_contact = contact

        return num_contacts

class EvalModel_Symmetry():
    
    def __init__(self, csv_file_path = any, save_path = any, time_to_ignore = 1):
        
        df = pd.read_csv(csv_file_path)
        self.full_data = df
        self.save_path = save_path
        # 102 corresponds to the first second of data which is normally when the robot is still accelerating
        self.num_points_to_ignore = time_to_ignore*100 + 2
        
        indices = []
    
        # separate data into different episodes 
        for index, row in self.full_data.iterrows():
            if row[0] == 0 and index != 0: 
                indices.append(index)
        indices.append(self.full_data.index[-1])
        
        self.num_episode = len(indices)
        self.epiosde_data = []
        
        for i in range(len(indices) - 1):
            start = indices[i]
            end = indices[i+1]
            df_temp = self.full_data.iloc[start:end]
            self.epiosde_data.append(df_temp)
        
        for j in range(self.num_episode - 1):
            self.episode_save_path = "Episode " + str(j) + "/"
            # make paths as neccessary
            os.makedirs((self.save_path + self.episode_save_path + "Eval Nums/Plots/"), exist_ok=True)
            os.makedirs((self.save_path + self.episode_save_path + "Eval Nums/Values/"), exist_ok=True)

        # Plot data for each episode
        for i in range(self.num_episode - 1):

            self.foot_contact_data = []
            
            # All the files to execute
            self.check_angle_symmetry(i)
            self.check_torque_symmetry(i)
            self.check_loading_symmetry(i)
            self.get_joint_angle_symmetry(i)
            self.get_foot_symmetry(i)   
            self.get_joint_torque_symmetry(i)
            #self.num_contacts(i)
            #self.write_to_symmetry_csv(i) 
            self.get_max_symmetry(i) 
            self.make_heatmaps(i) 
            
    
    def check_angle_symmetry(self, i):
        # Check if the angles are symmetric 
        
        headings = ["FR Hip Ad_Ab Angle", "FR Hip Swing Angle", "FR Knee Angle",
                    "FL Hip Ad_Ab Angle", "FL Hip Swing Angle", "FL Knee Angle",
                    "BR Hip Ad_Ab Angle", "BR Hip Swing Angle", "BR Knee Angle",
                    "BL Hip Ad_Ab Angle", "BL Hip Swing Angle", "BL Knee Angle"]
        
        angle_headings = ["Hip", "Thigh", "Calf"]
        leg_headings = ["FR", "FL", "BR", "BL"]
        adjusted_headings =["FR Hip Ad_Ab Angle", "FL Hip Ad_Ab Angle", "BR Hip Ad_Ab Angle", "BL Hip Ad_Ab Angle", 
                            "FR Hip Swing Angle", "FL Hip Swing Angle", "BR Hip Swing Angle",  "BL Hip Swing Angle", 
                            "FR Knee Angle", "FL Knee Angle", "BR Knee Angle", "BL Knee Angle"]
        
        angle_limits = [[-45, 45], [-60, 240], [-154.5, -52.5]]
        self.angle_symmetry_norm = {}
        self.angle_symmetry_norm_max = []
        self.angle_symmetry_norm_min = []

        self.angle_symmetry = {}
        for j in range(12):
            data = changeRadtoDegrees(self.epiosde_data[i][headings[j]])
            self.angle_symmetry[headings[j] + " max"] = data.max()
            self.angle_symmetry[headings[j] + " min"] = data.min()
            self.angle_symmetry[headings[j] + " difference"] = data.max() - data.min()  
        
        for l in range(3):
            for m in range(4):
                values = changeRadtoDegrees(self.epiosde_data[i][adjusted_headings[m + l*4]])
                self.angle_symmetry_norm_max.append(get_percentage_of_range(values.max(), angle_limits[l][0], angle_limits[l][1]))
                self.angle_symmetry_norm_min.append(get_percentage_of_range(values.min(), angle_limits[l][0], angle_limits[l][1]))


    def check_torque_symmetry(self, i):
        # Check if the angles are symmetric 
        
        headings = ["FR Hip Torque", "FR Knee Torque", "FR Ankle Torque",
                    "FL Hip Torque", "FL Knee Torque", "FL Ankle Torque",
                    "BR Hip Torque", "BR Knee Torque", "BR Ankle Torque",
                    "BL Hip Torque", "BL Knee Torque", "BL Ankle Torque"]
        self.torque_symmetry = {}
        adjusted_headings =["FR Hip Torque", "FL Hip Torque", "BR Hip Torque", "BL Hip Torque", 
                            "FR Knee Torque", "FL Knee Torque", "BR Knee Torque",  "BL Knee Torque", 
                            "FR Ankle Torque", "FL Ankle Torque", "BR Ankle Torque", "BL Ankle Torque"]
        torque_limits = [-35, 35]
        self.torque_symmetry_norm_max = []
        self.torque_symmetry_norm_min = []
        

        for j in range(12):
            data = self.epiosde_data[i][headings[j]]
            self.torque_symmetry[headings[j] + " max"] = data.max()
            self.torque_symmetry[headings[j] + " min"] = data.min()
            self.torque_symmetry[headings[j] + " difference"] = data.max() - data.min()
        
        for l in range(3):
            for m in range(4):
                values = self.epiosde_data[i][adjusted_headings[m + l*4]]
                self.torque_symmetry_norm_max.append(get_percentage_of_range(values.max(), torque_limits[0], torque_limits[1]))
                self.torque_symmetry_norm_min.append(get_percentage_of_range(values.min(), torque_limits[0], torque_limits[1]))

    def check_loading_symmetry(self, i):
        # Check if the angles are symmetric 
        
        data = ["FR GRF", "FL GRF", "BR GRF", "BL GRF"]
        contacts = ["FR Contact", "FL Contact", "BR Contact", "BL Contact"]

        self.GRF_symmetry = {}
        GRF_max = []
        GRF_avg = []

        for j in range(len(contacts)):

            contact_data = self.epiosde_data[i][contacts[j]].values
            grf_data = self.epiosde_data[i][data[j]].values

            summed_data = 0
            num_ones = 0
            stored_data =[]

            for n in range(len(contact_data)):    
                # start contact 
                if contact_data[n] == 1:
                    summed_data += grf_data[n]
                    num_ones += 1
                # end contact 
                elif contact_data[n] == 0 and num_ones > 0:
                    stored_data.append(summed_data / (num_ones * 0.01))
                    # Do something with the result
                    summed_data = 0
                    num_ones = 0
            
            self.GRF_symmetry[data[j] + " max"] = np.max(stored_data)
            self.GRF_symmetry[data[j] + " avg"] = np.mean(stored_data)
            GRF_max.append(np.max(stored_data))
            GRF_avg.append(np.mean(stored_data))
            
        self.GFR_max_percenatge = []
        self.GRF_avg_percentage = []

        for j in range(4):
            self.GFR_max_percenatge.append((GRF_max[j] / np.sum(GRF_max)) * 100)
            self.GRF_avg_percentage.append((GRF_avg[j] / np.sum(GRF_avg)) * 100)    
        
        
    def get_joint_angle_symmetry(self, i):
    
        self.hip_headings = ["FR Hip Ad_Ab Angle", "FL Hip Ad_Ab Angle", "BR Hip Ad_Ab Angle", "BL Hip Ad_Ab Angle"]
        self.thigh_headings = ["FR Hip Swing Angle", "FL Hip Swing Angle", "BR Hip Swing Angle", "BL Hip Swing Angle"]
        self.calf_headings = ["FR Knee Angle", "FL Knee Angle", "BR Knee Angle", "BL Knee Angle"]
    
        self.spearman_hip_angles_matrix = np.zeros((len(self.hip_headings), len(self.hip_headings)))
        self.spearman_thigh_angles_matrix = np.zeros((len(self.thigh_headings), len(self.thigh_headings)))
        self.spearman_calf_angles_matrix = np.zeros((len(self.calf_headings), len(self.calf_headings)))
        spearman_hip_angles_matrix = np.zeros((len(self.hip_headings), len(self.hip_headings)))
        cc_score_hip_angles_matrix = np.zeros((len(self.hip_headings), len(self.hip_headings)))
        spearman_thigh_angles_matrix = np.zeros((len(self.thigh_headings), len(self.thigh_headings)))
        cc_score_thigh_angles_matrix = np.zeros((len(self.thigh_headings), len(self.thigh_headings)))
        spearman_calf_angles_matrix = np.zeros((len(self.calf_headings), len(self.calf_headings)))
        cc_score_calf_angles_matrix = np.zeros((len(self.calf_headings), len(self.calf_headings)))
    
        for l in range(3):
            
            if l == 0:
                joint_headings = self.hip_headings
                spearman_joint_angles_matrix = spearman_hip_angles_matrix
                cc_score_joint_angles_matrix = cc_score_hip_angles_matrix
            elif l == 1: 
                joint_headings = self.thigh_headings
                spearman_joint_angles_matrix = spearman_thigh_angles_matrix
                cc_score_joint_angles_matrix = cc_score_thigh_angles_matrix
            elif l == 2:  
                joint_headings = self.calf_headings
                spearman_joint_angles_matrix = spearman_calf_angles_matrix
                cc_score_joint_angles_matrix = cc_score_calf_angles_matrix
    
            for j in range(len(self.hip_headings)):
    
                for k in range(len(self.hip_headings)):
    
                    data1 = changeRadtoDegrees(self.epiosde_data[i][joint_headings[j]])
                    data2 = changeRadtoDegrees(self.epiosde_data[i][joint_headings[k]])
                    spearman_corr, _ = spearmanr(data1, data2)
                    tau, _ = kendalltau(data1, data2)
                    spearman_joint_angles_matrix[j][k] = spearman_corr
                    cc_score_joint_angles_matrix[j][k] = tau
            
            if l == 0:
                self.spearman_hip_angles_matrix = spearman_hip_angles_matrix
                self.cc_score_hip_angles_matrix = cc_score_hip_angles_matrix
            elif l == 1: 
                self.spearman_thigh_angles_matrix = spearman_thigh_angles_matrix
                self.cc_score_thigh_angles_matrix = cc_score_thigh_angles_matrix
            elif l == 2:  
                self.spearman_calf_angles_matrix = spearman_calf_angles_matrix
                self.cc_score_calf_angles_matrix = cc_score_calf_angles_matrix

    def get_foot_symmetry(self, i):
        self.foot_headings = ["FR Foot Position Z", "FL Foot Position Z", "BR Foot Position Z", "BL Foot Position Z"]
        headings = ["FR", "FL", "BR", "BL"]

        self.spearman_foot_z_matrix = np.zeros((len(self.foot_headings), len(self.foot_headings)))
        self.cc_score_foot_z_matrix = np.zeros((len(self.foot_headings), len(self.foot_headings)))

        max_min_values = {}
        max_s, text_max_s = 0, None
        min_s, text_min_s = 1000, None
        max_t, text_max_t = 0, None
        min_t, text_min_t = 1000, None

        for j in range(len(self.foot_headings)):

            for k in range(len(self.foot_headings)):
                
                data1 = self.epiosde_data[i][self.foot_headings[j]]
                data2 = self.epiosde_data[i][self.foot_headings[k]]
                spearman_corr, _ = spearmanr(data1, data2)
                tau, _ = kendalltau(data1, data2)
                self.spearman_foot_z_matrix[j][k] = spearman_corr
                self.cc_score_foot_z_matrix[j][k] = tau
        
                if spearman_corr != 1 and tau != 1:
                    if spearman_corr > max_s:
                        max_s = spearman_corr
                        text_max_s = f"{headings[j]} {headings[k]}"
                    if spearman_corr < min_s:
                        min_s = spearman_corr
                        text_min_s = f"{headings[j]} {headings[k]}"
                    if tau > max_t:
                        max_t = tau
                        text_max_t = f"{headings[j]} {headings[k]}"
                    if tau < min_t:
                        min_t = tau
                        text_min_t = f"{headings[j]} {headings[k]}"

        max_min_values["Max Spearman"] = [max_s, text_max_s]
        max_min_values["Min Spearman"] = [min_s, text_min_s]
        max_min_values["Max Kendall"] = [max_t, text_max_t]
        max_min_values["Min Kendall"] = [min_t, text_min_t]
    
    def get_joint_torque_symmetry(self, i):
    
        self.hip_headings_torque = ["FR Hip Torque", "FL Hip Torque", "BR Hip Torque", "BL Hip Torque"]
        self.thigh_headings_torque = ["FR Knee Torque", "FL Knee Torque", "BR Knee Torque", "BL Knee Torque"]
        self.calf_headings_torque = ["FR Ankle Torque", "FL Ankle Torque", "BR Ankle Torque", "BL Ankle Torque"]
    
        self.spearman_hip_torque_matrix = np.zeros((len(self.hip_headings_torque), len(self.hip_headings_torque)))
        self.spearman_thigh_torques_matrix = np.zeros((len(self.thigh_headings_torque), len(self.thigh_headings_torque)))
        self.spearman_calf_torques_matrix = np.zeros((len(self.calf_headings_torque), len(self.calf_headings_torque)))
        spearman_hip_torques_matrix = np.zeros((len(self.hip_headings_torque), len(self.hip_headings_torque)))
        cc_score_hip_torques_matrix = np.zeros((len(self.hip_headings_torque), len(self.hip_headings_torque)))
        spearman_thigh_torques_matrix = np.zeros((len(self.thigh_headings_torque), len(self.thigh_headings_torque)))
        cc_score_thigh_torques_matrix = np.zeros((len(self.thigh_headings_torque), len(self.thigh_headings_torque)))
        spearman_calf_torques_matrix = np.zeros((len(self.calf_headings_torque), len(self.calf_headings_torque)))
        cc_score_calf_torques_matrix = np.zeros((len(self.calf_headings_torque), len(self.calf_headings_torque)))
    
        for l in range(3):
            
            if l == 0:
                joint_headings = self.hip_headings_torque
                spearman_joint_torques_matrix = spearman_hip_torques_matrix
                cc_score_joint_torques_matrix = cc_score_hip_torques_matrix
            elif l == 1: 
                joint_headings = self.thigh_headings_torque
                spearman_joint_torques_matrix = spearman_thigh_torques_matrix
                cc_score_joint_torques_matrix = cc_score_thigh_torques_matrix
            elif l == 2:  
                joint_headings = self.calf_headings_torque
                spearman_joint_torques_matrix = spearman_calf_torques_matrix
                cc_score_joint_torques_matrix = cc_score_calf_torques_matrix
    
            for j in range(len(self.hip_headings_torque)):
    
                for k in range(len(self.hip_headings_torque)):
    
                    data1 = self.epiosde_data[i][joint_headings[j]]
                    data2 = self.epiosde_data[i][joint_headings[k]]
                    spearman_corr, _ = spearmanr(data1, data2)
                    tau, _ = kendalltau(data1, data2)
                    spearman_joint_torques_matrix[j][k] = spearman_corr
                    cc_score_joint_torques_matrix[j][k] = tau
            
            if l == 0:
                self.spearman_hip_torque_matrix = spearman_joint_torques_matrix
                self.cc_score_hip_torque_matrix = cc_score_joint_torques_matrix
            elif l == 1: 
                self.spearman_thigh_torques_matrix = spearman_joint_torques_matrix
                self.cc_score_thigh_torque_matrix = cc_score_joint_torques_matrix
            elif l == 2:  
                self.spearman_calf_torques_matrix = spearman_joint_torques_matrix
                self.cc_score_calf_torque_matrix = cc_score_joint_torques_matrix

    def num_contacts(self, i):
        #determine length of contact for each foot

        x_data = ["FR Contact", "FL Contact", "BR Contact", "BL Contact"]
        self.contact_headings = x_data
        self.contact_num = {}
        self.contact_lengths = {}   

        for j in range(len(x_data)):

            data = self.epiosde_data[i][x_data[j]]
            num_contacts, contact_lengths = self.count_contacts(data)
            self.contact_num[x_data[j] + " num contacts"] = num_contacts
            self.contact_lengths[x_data[j] + " contact lengths max and min and num min"] = [np.max(contact_lengths), np.min(contact_lengths), 
                                                                                            np.count_nonzero(contact_lengths == np.min(contact_lengths))]
        
    def write_to_symmetry_csv(self, i): 
        # Write all the data to a csv file
        with open(self.save_path + "Episode " + str(i) + "/Eval Nums/Values/symmetry_values.csv", mode='w', newline='') as file:
            writer = csv.writer(file)

            writer.writerow(["Joint Angle Symmetry", self.angle_symmetry])
            writer.writerow(["Joint Torque Symmetry", self.torque_symmetry])
            writer.writerow(["Leg loading Symmetry", self.GRF_symmetry])
            writer.writerow([ "Hip Angle Scores", self.hip_headings])
            writer.writerow(["Spearman", self.spearman_hip_angles_matrix])
            writer.writerow(["Kendall", self.cc_score_hip_angles_matrix])
            writer.writerow(["Thigh Angle Scores", self.thigh_headings])
            writer.writerow(["Spearman", self.spearman_thigh_angles_matrix])
            writer.writerow(["Kendall", self.cc_score_thigh_angles_matrix])
            writer.writerow(["Calf Angle Scores", self.calf_headings])
            writer.writerow(["Spearman", self.spearman_calf_angles_matrix])
            writer.writerow(["Kendall", self.cc_score_calf_angles_matrix])
            writer.writerow(["Foot z-pos Scores", self.foot_headings])
            writer.writerow(["Foot Symmetry", self.spearman_foot_z_matrix])
            writer.writerow(["Kendall", self.cc_score_foot_z_matrix])
            writer.writerow(["Hip torque Scores", self.hip_headings_torque])
            writer.writerow(["Spearman", self.spearman_hip_torque_matrix])
            writer.writerow(["Kendall", self.cc_score_hip_torque_matrix])
            writer.writerow(["Thigh Torque Scores", self.thigh_headings_torque])
            writer.writerow(["Spearman", self.spearman_thigh_torques_matrix])
            writer.writerow(["Kendall", self.cc_score_thigh_torque_matrix])
            writer.writerow(["Calf Torque Scores", self.calf_headings_torque])
            writer.writerow(["Spearman", self.spearman_calf_torques_matrix])
            writer.writerow(["Kendall", self.cc_score_calf_torque_matrix])
            writer.writerow(["", self.contact_headings])
            writer.writerow(["Number of Contacts", self.contact_num, self.contact_lengths])    
    
    def make_heatmaps(self, i):

        # First make the foot symmetry heatmap
        headings = ["FR", "FL", "BR", "BL"]
        row_names_1 = ["% Of max angle", "% Of min angle", "% Of max torque", "% Of min torque", "Max GRF", "Average GRF"]
        row_names_2 = ["Spearman Joint Angles", "Kendall Joint Angles", "Spearman Joint Torques", "Kendall Joint Torques", "Spearman Foot Z", "Kendall Foot Z"]
        col_names_2 = ["FR + BL", "FR + BR", " FL + BL", " BL + BR", "FR + FL", "FL + BR"]
        plots = ["Hip", "Thigh", "Calf"]
        
        for j in range(3):
            fig, ax = plt.subplots(1, 2, figsize=(18, 8))
            
            if j == 0: 
                data1 = self.hip_angles_spearman
                data2 = self.hip_angles_tau
                data3 = self.hip_torque_spearman
                data4 = self.hip_torque_tau
                data5 = self.foot_spearman
                data6 = self.foot_tau   
                data2_2d = np.vstack((data1, data2, data3, data4, data5, data6))
                
            elif j == 1:
                data1 = self.thigh_angles_spearman
                data2 = self.thigh_angles_tau
                data3 = self.thigh_torque_spearman
                data4 = self.thigh_torque_tau
                data5 = self.foot_spearman
                data6 = self.foot_tau
                data2_2d = np.vstack((data1, data2, data3, data4, data5, data6))
                
            elif j == 2:
                data1 = self.calf_angles_spearman
                data2 = self.calf_angles_tau
                data3 = self.calf_torque_spearman
                data4 = self.calf_torque_tau
                data5 = self.foot_spearman
                data6 = self.foot_tau
                data2_2d = np.vstack((data1, data2, data3, data4, data5, data6))
        
            data1 = self.angle_symmetry_norm_max[j*4:j*4+4]
            data2 = self.angle_symmetry_norm_min[j*4:j*4+4]
            data3 = self.torque_symmetry_norm_max[j*4:j*4+4]
            data4 = self.torque_symmetry_norm_min[j*4:j*4+4]
            data5 = self.GFR_max_percenatge
            data6 = self.GRF_avg_percentage
            
            data_2d = np.vstack((data1, data2, data3, data4, data5, data6))

            cbar_kw = {"shrink": .9}
            im1, cbar = heatmap(data_2d, row_names_1, headings, ax=ax[0],
                       cmap="twilight_shifted", vmin=0, vmax=100, cbarlabel="% Values", cbar_kw=cbar_kw)
            cbar.ax.tick_params(labelsize=13)
            texts = annotate_heatmap(im1, valfmt="{x:.2f}", size=12, textcolors=("black", "black"))

            im2, cbar = heatmap(data2_2d, row_names_2, col_names_2, ax=ax[1],
                       cmap="Purples", vmin=-1, vmax=1, cbarlabel="", cbar_kw=cbar_kw)
            cbar.ax.tick_params(labelsize=13)  # Set the size of the colorbar labels
            texts = annotate_heatmap(im2, valfmt="{x:.3f}", size=12, textcolors=("black", "black"))
            fig.tight_layout(pad=3.0)

            fig.suptitle(plots[j] + " Symmetry Heatmap", fontsize=16)
            plt.savefig(self.save_path + "Episode " + str(i) + "/Eval Nums/Plots/9a " + plots[j] + " Heatmaps.png", bbox_inches="tight")
            plt.clf()
 
    def get_max_symmetry(self, i):

        for l in range(3):
            
            if l == 0:
                data_1 = self.spearman_hip_angles_matrix 
                data_2 = self.cc_score_hip_angles_matrix
                data_3 = self.spearman_hip_torque_matrix
                data_4 = self.cc_score_hip_torque_matrix
                
                unique_values_data_1 = np.unique(data_1)
                unique_values_data_2 = np.unique(data_2)
                unique_values_data_3 = np.unique(data_3)
                unique_values_data_4 = np.unique(data_4)
                
                self.hip_angles_spearman = unique_values_data_1[:-1]
                self.hip_angles_tau = unique_values_data_2[:-1]
                self.hip_torque_spearman = unique_values_data_3[:-1]
                self.hip_torque_tau = unique_values_data_4[:-1]
                
            elif l == 1: 
                data_1 = self.spearman_thigh_angles_matrix
                data_2 = self.cc_score_thigh_angles_matrix
                data_3 = self.spearman_thigh_torques_matrix
                data_4 = self.cc_score_thigh_torque_matrix
                
                unique_values_data_1 = np.unique(data_1)
                unique_values_data_2 = np.unique(data_2)
                unique_values_data_3 = np.unique(data_3)
                unique_values_data_4 = np.unique(data_4)
                
                self.thigh_angles_spearman = unique_values_data_1[:-1]
                self.thigh_angles_tau = unique_values_data_2[:-1]
                self.thigh_torque_spearman = unique_values_data_3[:-1]
                self.thigh_torque_tau = unique_values_data_4[:-1]
                
            elif l == 2:  
                data_1 = np.round(self.spearman_calf_angles_matrix, decimals=6)
                data_2 = np.round(self.cc_score_calf_angles_matrix, decimals=6)
                data_3 = np.round(self.spearman_calf_torques_matrix, decimals=6)
                data_4 = np.round(self.cc_score_calf_torque_matrix, decimals=6)
                
                unique_values_data_1 = np.unique(data_1)
                unique_values_data_2 = np.unique(data_2)
                unique_values_data_3 = np.unique(data_3)
                unique_values_data_4 = np.unique(data_4)
                
                self.calf_angles_spearman = unique_values_data_1[:-1]   
                self.calf_angles_tau = unique_values_data_2[:-1]
                self.calf_torque_spearman = unique_values_data_3[:-1]
                self.calf_torque_tau = unique_values_data_4[:-1]
        
        self.foot_spearman = np.unique(np.round(self.spearman_foot_z_matrix, 5))[:-1]
        self.foot_spearman = [self.foot_spearman[4], self.foot_spearman[5], self.foot_spearman[3], self.foot_spearman[2], self.foot_spearman[1], self.foot_spearman[0]]
        self.foot_tau = np.unique(np.round(self.cc_score_foot_z_matrix, 5))[:-1]
        self.foot_tau  = [self.foot_tau[4], self.foot_tau[5], self.foot_tau[3], self.foot_tau[2], self.foot_tau[1], self.foot_tau[0]]
            

# very useful functions
def changeRadtoDegrees(data):
    return data * 180 / np.pi  

def count_contacts(data):
    num_contacts = 0
    prev_contact = 0
    contact_len = 0
    # store the legnth of each contact
    contact_lengths = []

    for contact in data:
        if contact == 1: 
            if prev_contact == 0:
                num_contacts += 1
                contact_len += 0.01
            elif prev_contact == 1:
                contact_len += 0.01
        elif contact == 0 and prev_contact == 1:
            contact_lengths.append(contact_len)
            contact_len = 0
        prev_contact = contact

    # Check if the last item is 1
    if prev_contact == 1:
        contact_lengths.append(contact_len)

    return num_contacts, contact_lengths
    
def normalise_between_neg_pos_1(value, min_value, max_value):
    return ((2 * (value - min_value)) / (max_value - min_value)) - 1

def get_percentage_of_range(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value) * 100

# used for curve fitting 
def sinusoidal(x, A, omega, phi, offset):
    return A * np.sin(omega * x + phi) + offset

def piecewise_sine(x, amplitude, frequency, phase):
    sine_value = (amplitude * np.sin(frequency * x + phase)) + np.abs((amplitude * np.sin(frequency * x + phase)))
    return sine_value

def positive_sine(x, amplitude, frequency, phase):
    sine_value = np.abs(amplitude * np.sin(frequency * x + phase))
    return sine_value

def linear(x, m, b):
    return m * x + b

# Heatmap plotting functions - from https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    try:
        ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
        ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)
    except AttributeError:
        ax.set_xticks(np.arange(len(data)), labels=col_labels)
        ax.set_yticks(np.arange(len(data)), labels=row_labels)
    

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)
    try: 
        ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)
    except AttributeError:
        ax.set_xticks(np.arange(len(data)+1)-.5, minor=True)
        ax.set_yticks(np.arange(len(data)+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    try:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)
    except AttributeError:
        for i in range(len(data)):
            for j in range(len(data)):
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)
                
    return texts
    # '''
    # Heatmap Colours ===
    # 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 
    # 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 
    # 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 
    # 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 
    # 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 
    # 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 
    # 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 
    # 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 
    # 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 
    # 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 
    # 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 
    # 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 
    # 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 
    # 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 
    # 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 
    # 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 
    # 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 
    # 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 
    # 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 
    # 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 
    # 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 
    # 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'
    # '''

def heatmap_columns(data, row_labels, col_labels, ax=None,
                    cbar_kw=None, cbarlabel="", color_columns=None, color_rows=None, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted. If
        not provided, use current Axes or create a new one. Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`. Optional.
    cbarlabel
        The label for the colorbar. Optional.
    color_columns
        A list of column indices to be colored. Optional.
    color_rows
        A list of row indices to be colored. Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    if color_columns is None:
        color_columns = []
    if color_rows is None:
        color_rows = []

    # Create a mask for the data array to only show specified columns and rows
    mask = np.ones_like(data, dtype=bool)
    if color_columns:
        mask[:, color_columns] = False
    if color_rows:
        mask[color_rows, :] = False

    # Create a masked array to plot only the specified columns and rows
    data_masked = np.ma.masked_array(data, mask)

    # Plot the masked heatmap
    im = ax.imshow(data_masked, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create gray grid only on specified rows and columns.
    ax.spines[:].set_visible(False)

    # Set minor ticks to create a grid
    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    
    # Create the grid
    ax.grid(which="minor", color="lightgray", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Ensure grid lines cover the entire plot area
    ax.set_xlim(-0.5, data.shape[1]-0.5)
    ax.set_ylim(data.shape[0]-0.5, -0.5)

    # Add bottom and right spines
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_color('lightgray')
    ax.spines['right'].set_color('lightgray')
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)

    # Make row labels bold
    ax.set_yticklabels(row_labels, fontweight='bold', fontsize=13)
    ax.set_xticklabels(col_labels, fontweight='bold', fontsize=13)

    return im, cbar

def annotate_heatmap_col(im, data=None, valfmt="{x:.2f}",
                         textcolors=("black", "white"),
                         threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate. If None, the image's data is used. Optional.
    valfmt
        The format of the annotations inside the heatmap. This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`. Optional.
    textcolors
        A pair of colors. The first is used for values below a threshold,
        the second for those above. Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied. If None (the default) uses the middle of the colormap as
        separation. Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

class FootContactPlots:
    
    def __init__(self, csv_file_path = any, save_path = any, time_to_ignore = 1):
        
        df = pd.read_csv(csv_file_path)
        self.full_data = df
        self.save_path = save_path
        # 102 corresponds to the first second of data which is normally when the robot is still accelerating
        self.num_points_to_ignore = time_to_ignore*100 + 2
        
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
            # make paths as neccessary
            os.makedirs((self.save_path + self.episode_save_path + "Eval Nums/0 Foot Plot/"), exist_ok=True)

        # Plot data for each episode
        for i in range(self.num_episode - 1):

            # All the files to plot 
            self.plot_foot_contact(i)
    
    def plot_foot_contact(self, i):
        # Same function as in evaluation.py 
        
        for i in range(self.num_episode - 1):
            plt.close()
        
            # y1, y1_data = self.findContactNumber(self.epiosde_data[i]["FR Contact"].values)
            # y2, y2_data = self.findContactNumber(self.epiosde_data[i]["FL Contact"].values)
            # y3, y3_data = self.findContactNumber(self.epiosde_data[i]["BR Contact"].values)
            # y4, y4_data = self.findContactNumber(self.epiosde_data[i]["BL Contact"].values)
            y1_data = self.findContact(self.epiosde_data[i]["FR Foot Position Z"][self.num_points_to_ignore:402].values, filter = 0.025)
            y2_data = self.findContact(self.epiosde_data[i]["FL Foot Position Z"][self.num_points_to_ignore:402].values)
            y3_data = self.findContact(self.epiosde_data[i]["BR Foot Position Z"][self.num_points_to_ignore:402].values)
            y4_data = self.findContact(self.epiosde_data[i]["BL Foot Position Z"][self.num_points_to_ignore:402].values)
            
            testdict = {}
            testdict["FR"] = y1_data
            testdict["FL"] = y2_data
            testdict["BR"] = y3_data
            testdict["BL"] = y4_data
            
            fig, ax = plt.subplots()
            ax = self.plotbooleans(ax, testdict)
            
            #plt.xlabel("Episode Time")
            plt.title(("Foot Contact Patterns"))
            
            ax = plt.gca()
            x_lim = ax.get_xlim()
            ylim = ax.get_ylim()
            plt.subplots_adjust(bottom=0.2)
            plt.savefig(self.save_path + "Episode " + str(i) + "/Eval Nums/0 Foot Plot/" + "01 Foot Contacts.pdf")
            plt.clf()
            
    
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
        
    def findones(self, a): 
        # functions from https://stackoverflow.com/questions/49634844/matplotlib-boolean-plot-rectangle-fill 
        isone = np.concatenate(([0], a, [0]))
        absdiff = np.abs(np.diff(isone))
        ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
        return np.clip(ranges, 0, len(a) - 1)
    
    
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
    
    def findContact(self, data, filter = 0.028):
        contact = []
        for value in data:
            if value <= filter:
                contact.append(1)
            else:
                contact.append(0)
        return contact
    