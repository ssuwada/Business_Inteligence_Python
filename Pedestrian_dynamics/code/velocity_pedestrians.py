#   ---------PROJECT CROSSING ANGLE-----------
#                Report task
#             till 20th of May
#           ---------------------
#              SEBASTIAN SUWADA 
#           ---------------------
#           Velocity calculation part

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import glob
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.ticker as ticker
import os



def read_all_csv():

# Define the path to the directory containing the CSV files
    path = '/Users/sebastiansuwada/Desktop/Python_Practice/Python_Project_BI/Pedestrian_crossing/crossing_flows_data/*.csv'

# Use the glob function to create a list of file names that match a pattern
# For example, all files with the extension .csv in the specified directory
    all_files = glob.glob(path)

# Create an empty list to store the dataframes
    list_of_dataframes = []
    file_names = []

# Iterate over the list of file names and read each file into a pandas dataframe
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        list_of_dataframes.append(df)
        file_names.append(os.path.basename(filename))
                          
    #print(file_names)

    return list_of_dataframes, file_names

def k_means(xy_stack2D):
    kmeans1 = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans1.fit(xy_stack2D)
    return kmeans1.labels_

def mean(lst):
    return sum(lst)/len(lst)

def calculate_angle(all_csv, file_names):

    n = 1 
    n2 = 2
    k_means_temp_init = []
    k_means_temp_final = []
    collect_angle_degree = []

    groupinit1x = []
    groupinit1y = []
    groupinit2x = []
    groupinit2y = []

    groupfinal1x = []
    groupfinal1y = []
    groupfinal2x = []
    groupfinal2y = []

    collect_angle_degree.clear()

    for i in range(len(all_csv)):


        #   Initial values
        #   In this data set we have two groups in one variable!
        temp1x = all_csv[i].iloc[1, n::(2*n)].to_numpy()    
        temp1y = all_csv[i].iloc[1, n2::(2*n)].to_numpy()

        #   End values
        temp2x = all_csv[i].iloc[-1, n::(2*n)].to_numpy()
        temp2y = all_csv[i].iloc[-1, n2::(2*n)].to_numpy()

        combined_arr_init1xy = np.column_stack((temp1x, temp1y))
        combined_arr_init2xy = np.column_stack((temp2x, temp2y))

        #   PERFORM CLUSTERING FIRST INITIAL GROUPS

        # cluster each set of data using K-means
        k_means_temp_init.append(k_means(combined_arr_init1xy))
        # chceck score of labels for clusters:

        #   PERFORM CLUSTERING FIRST FINAL GROUPS

        # cluster each set of data using K-means
        k_means_temp_final.append(k_means(combined_arr_init2xy))
        # chceck score of labels for clusters:

        #   PERFORM LOOP FOR CHOOSE AND DEFINE NEW GROUPS FROM CLUSTERING

        #   INITIAL VALUES 

        for l in range(len(k_means_temp_init[i])):
            if k_means_temp_init[i][l] == 0:
                groupinit1x.append(temp1x[l])
                groupinit1y.append(temp1y[l])
            if k_means_temp_init[i][l] == 1:
                groupinit2x.append(temp1x[l])
                groupinit2y.append(temp1y[l])

        # FINAL VALUES

        for l1 in range(len(k_means_temp_init[i])):
            if k_means_temp_init[i][l1] == 0:
                groupfinal1x.append(temp2x[l1])
                groupfinal1y.append(temp2y[l1])
            if k_means_temp_init[i][l1] == 1:
                groupfinal2x.append(temp2x[l1])
                groupfinal2y.append(temp2y[l1])

        group1_initial_barycenter = np.array([mean(groupinit1x), mean(groupinit1y)])
        group1_final_barycenter = np.array([mean(groupfinal1x), mean(groupfinal1y)])
        group2_initial_barycenter = np.array([mean(groupinit2x), mean(groupinit2y)])
        group2_final_barycenter = np.array([mean(groupfinal2x), mean(groupfinal2y)])

        #   Motion of each group
        group1_motion_direction = group1_final_barycenter - group1_initial_barycenter
        group2_motion_direction = group2_final_barycenter - group2_initial_barycenter

        # Calculate the intersection point of the two motion direction lines
        M = np.column_stack((group1_motion_direction, group2_motion_direction))
        v = group2_initial_barycenter - group1_initial_barycenter
        t = np.linalg.solve(M, v)
        intersection_point = group1_initial_barycenter + t[0] * group1_motion_direction

        # Calculate the crossing angle between the two groups
        group1_to_intersection = intersection_point - group1_initial_barycenter
        group2_to_intersection = intersection_point - group2_initial_barycenter
        crossing_angle = np.arccos(np.dot(group1_to_intersection, group2_to_intersection) / (np.linalg.norm(group1_to_intersection) * np.linalg.norm(group2_to_intersection)))
        crossing_angle_degrees = np.degrees(crossing_angle)

        collect_angle_degree.append(crossing_angle_degrees)

        groupinit1x.clear()
        groupinit1y.clear()
        groupinit2x.clear()
        groupinit2y.clear()

        groupfinal1x.clear()
        groupfinal1y.clear()
        groupfinal2x.clear()
        groupfinal2y.clear()
    
    return collect_angle_degree

def assign_to_angle_groups(file_names, cross_ang):

    # Define theoretical value on calculated data set

    # CREATE A LOOP WHICH IS CHECKIN THOSE VALUE AND DEFINE THEM INTO THEORY VALUE:

    degree_0_ang = []
    degree_0_name = []
    degree_30_ang = []
    degree_30_name = []
    degree_60_ang = []
    degree_60_name = []
    degree_90_ang = []
    degree_90_name = []
    degree_120_ang = []
    degree_120_name = []
    degree_150_ang = []
    degree_150_name = []

    comb_0 = []
    comb_30 = []
    comb_60 = []
    comb_90 = []
    comb_120 = []
    comb_150 = []

    cross_assign = 0

    for i in range(len(cross_ang)):
        # 0 degree
        if cross_ang[i] < 10:
            degree_0_ang.append(cross_ang[i])
            degree_0_name.append(file_names[i])
            cross_assign = 0

        # 30 degree
        if cross_ang[i] < 40 and cross_ang[i] > 20:
            degree_30_ang.append(cross_ang[i])
            degree_30_name.append(file_names[i])
            cross_assign = 30

        # 60 degree
        if cross_ang[i] < 70 and cross_ang[i] > 50:
            degree_60_ang.append(cross_ang[i])
            degree_60_name.append(file_names[i])
            cross_assign = 60

        # 90 degree
        if cross_ang[i] < 100 and cross_ang[i] > 80:
            degree_90_ang.append(cross_ang[i])
            degree_90_name.append(file_names[i])
            cross_assign = 90

        # 120 degree
        if cross_ang[i] < 130 and cross_ang[i] > 110:
            degree_120_ang.append(cross_ang[i])
            degree_120_name.append(file_names[i])
            cross_assign = 120

        # 150 degree
        if cross_ang[i] < 160 and cross_ang[i] > 140:
            degree_150_ang.append(cross_ang[i])
            degree_150_name.append(file_names[i])
            cross_assign = 150


        # 180 degree - same as 0 degree
    comb_0 = np.column_stack((degree_0_name, degree_0_ang))
    comb_30 = np.column_stack((degree_30_name, degree_30_ang))
    comb_60 = np.column_stack((degree_60_name, degree_60_ang))
    comb_90 = np.column_stack((degree_90_name, degree_90_ang))
    comb_120 = np.column_stack((degree_120_name, degree_120_ang))
    comb_150 = np.column_stack((degree_150_name, degree_150_ang))

    return comb_0, comb_30, comb_60, comb_90, comb_120, comb_150

def velocity(all_csv):
    n = 1
    i = 0

    assign_velocity = []
    time = []
    file_velocity = []
    agenr_file_vel = []
    file_velocity.clear()
    atemp = []

    for i in range(len(all_csv)):
        
        t = 1
        time.clear()
        #print((all_csv[i].shape[1]-1)/2)
        temp = (all_csv[i].shape[1]-1)/2

        #   Append time of every file
        for t in range(1,all_csv[i].shape[0]):
            time.append(t)
        agenr_file_vel.clear()

        #   For loop in range for amount of columns
        for n in range(int(temp)):
            #    Assign every value for x and y in every column for every row from one csv file
            assign_velocity.clear()
            temp_agent_x = all_csv[i].iloc[:, 2*n+1].to_numpy()    
            temp_agent_y = all_csv[i].iloc[:, 2*(n+1)].to_numpy()    
            # Velocity for one agent
            for tim in time:
                velocity = np.sqrt(np.power(temp_agent_x[tim]-temp_agent_x[tim-1],2)+np.power(temp_agent_y[tim]-temp_agent_y[tim-1],2))/1000 * 120
                
                assign_velocity.append(velocity)

            agenr_file_vel.extend(assign_velocity)

        atemp = list(agenr_file_vel)  # Create a copy of the list
        file_velocity.append(atemp)

    ret_val = list(file_velocity)#np.column_stack((file_names ,file_velocity))

    return ret_val

def plot_histogram(comb_0, comb_30, comb_60, comb_90, comb_120, comb_150, velocity_arr, file_names):

    pos_0 = []
    pos_30 = []
    pos_60 = []
    pos_90 = []
    pos_120 = []
    pos_150 = []
    #print(velocity_arr)
    for i in range(len(comb_0)):
        for l in range(len(velocity_arr)):
            if file_names[l] == comb_0[i][0]:
                for k in range(len(velocity_arr[l])):
                    pos_0.append(velocity_arr[l][k])
    for i in range(len(comb_30)):
        for l in range(len(velocity_arr)):
            if file_names[l] == comb_30[i][0]:
                for k in range(len(velocity_arr[l])):
                    pos_30.append(velocity_arr[l][k])
    for i in range(len(comb_60)):
        for l in range(len(velocity_arr)):
            if file_names[l] == comb_60[i][0]:
                for k in range(len(velocity_arr[l])):
                    pos_60.append(velocity_arr[l][k])
    for i in range(len(comb_90)):
        for l in range(len(velocity_arr)):
            if file_names[l] == comb_90[i][0]:
                for k in range(len(velocity_arr[l])):
                    pos_90.append(velocity_arr[l][k])
    for i in range(len(comb_120)):
        for l in range(len(velocity_arr)):
            if file_names[l] == comb_120[i][0]:
                for k in range(len(velocity_arr[l])):
                    pos_120.append(velocity_arr[l][k])
    for i in range(len(comb_150)):
        for l in range(len(velocity_arr)):
            if file_names[l] == comb_150[i][0]:
                for k in range(len(velocity_arr[l])):
                    pos_150.append(velocity_arr[l][k])

    print(pos_0[100], pos_30[100], pos_60[100], pos_90[100], pos_120[100], pos_150[100])

    # Open the file in write mode
    with open('pos_0.txt', 'w') as file:
    # Write each element of the list on a new line
        for item in pos_0:
            file.write(str(item) + '\n')

    # Open the file in write mode
    with open('pos_30.txt', 'w') as file:
    # Write each element of the list on a new line
        for item in pos_30:
            file.write(str(item) + '\n')

    # Open the file in write mode
    with open('pos_60.txt', 'w') as file:
    # Write each element of the list on a new line
        for item in pos_60:
            file.write(str(item) + '\n')

    # Open the file in write mode
    with open('pos_90.txt', 'w') as file:
    # Write each element of the list on a new line
        for item in pos_90:
            file.write(str(item) + '\n')

    # Open the file in write mode
    with open('pos_120.txt', 'w') as file:
    # Write each element of the list on a new line
        for item in pos_120:
            file.write(str(item) + '\n')

    # Open the file in write mode
    with open('pos_150.txt', 'w') as file:
    # Write each element of the list on a new line
        for item in pos_150:
            file.write(str(item) + '\n')


#--------------- MAIN ----------------


all_csv, file_names = read_all_csv()
calc_ang = calculate_angle(all_csv,file_names)
comb_0, comb_30, comb_60, comb_90, comb_120, comb_150 = assign_to_angle_groups(file_names, calc_ang)
velocity_arr = velocity(all_csv)
plot_histogram(comb_0, comb_30, comb_60, comb_90, comb_120, comb_150, velocity_arr, file_names)


    # for i in range(len(comb_0)):
    #     for l in range(len(velocity_arr)):
    #         if velocity_arr[l][0] == comb_0[i][0]:
    #             for k in range(1,len(velocity_arr[l])):
    #                 pos_0.append(velocity_arr[l][k])
    # for i in range(len(comb_30)):
    #     for l in range(len(velocity_arr)):
    #         if velocity_arr[l][0] == comb_30[i][0]:
    #             for k in range(1,len(velocity_arr[l])):
    #                 pos_30.append(velocity_arr[l][k])
    # for i in range(len(comb_60)):
    #     for l in range(len(velocity_arr)):
    #         if velocity_arr[l][0] == comb_60[i][0]:
    #             for k in range(1,len(velocity_arr[l])):
    #                 pos_60.append(velocity_arr[l][k])
    # for i in range(len(comb_90)):
    #     for l in range(len(velocity_arr)):
    #         if velocity_arr[l][0] == comb_90[i][0]:
    #             for k in range(1,len(velocity_arr[l])):
    #                 pos_90.append(velocity_arr[l][k])
    # for i in range(len(comb_120)):
    #     for l in range(len(velocity_arr)):
    #         if velocity_arr[l][0] == comb_120[i][0]:
    #             for k in range(1,len(velocity_arr[l])):
    #                 pos_120.append(velocity_arr[l][k])
    # for i in range(len(comb_150)):
    #     for l in range(len(velocity_arr)):
    #         if velocity_arr[l][0] == comb_150[i][0]:
    #             for k in range(1,len(velocity_arr[l])):
    #                 pos_150.append(velocity_arr[l][k])