#   ---------PROJECT CROSSING ANGLE-----------
#                Report task
#             till 20th of May
#           ---------------------
#              SEBASTIAN SUWADA 
#           ---------------------
#   Deviation of every agent calculation part

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import glob
from sklearn.cluster import KMeans
import os
import math


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
                          
    print(file_names)

    return list_of_dataframes, file_names

def k_means(xy_stack2D):
    kmeans1 = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans1.fit(xy_stack2D)
    return kmeans1.labels_

def mean(lst):
    return sum(lst)/len(lst)

def calculate_deviation_ang(assign_ang,comput_ang):
    deviation = assign_ang - comput_ang
    return deviation

def calculate_angle_deviat(all_csv, file_names):
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

    deviation_part1 = []
    ext_deviat = []
    assign_angl = 0

    for i in range(len(all_csv)):

        #   Initial values
        #   In this data set we have two groups in one variable!

        collect_angle_degree.clear()
        for l1 in range(1,all_csv[i].shape[1]):
            temp1x = all_csv[i].iloc[l1, n::(2*n)].to_numpy()    
            temp1y = all_csv[i].iloc[l1, n2::(2*n)].to_numpy()

        #   End values
            temp2x = all_csv[i].iloc[all_csv[i].shape[1]-l1+1, n::(2*n)].to_numpy()
            temp2y = all_csv[i].iloc[all_csv[i].shape[1]-l1+1, n2::(2*n)].to_numpy()

            combined_arr_init1xy = np.column_stack((temp1x, temp1y))
            combined_arr_init2xy = np.column_stack((temp2x, temp2y))

        #   PERFORM CLUSTERING FIRST INITIAL GROUPS

        # cluster each set of data using K-means
            k_means_temp_init.append(k_means(combined_arr_init1xy))

        #   PERFORM CLUSTERING FIRST FINAL GROUPS

        # cluster each set of data using K-means
            k_means_temp_final.append(k_means(combined_arr_init2xy))

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
            #A = np.array([M, v])
            if np.linalg.det(M) != 0:
                t = np.linalg.solve(M, v)
                intersection_point = group1_initial_barycenter + t[0] * group1_motion_direction

        # Calculate the crossing angle between the two groups
                group1_to_intersection = intersection_point - group1_initial_barycenter
                group2_to_intersection = intersection_point - group2_initial_barycenter
                crossing_angle = np.arccos(np.dot(group1_to_intersection, group2_to_intersection) / (np.linalg.norm(group1_to_intersection) * np.linalg.norm(group2_to_intersection)))
                crossing_angle_degrees = np.degrees(crossing_angle)

                if crossing_angle_degrees < 10:
                    assign_angl = 0
        # 30 degree
                if crossing_angle_degrees < 40 and crossing_angle_degrees > 20:
                    assign_angl = 30
        # 60 degree
                if crossing_angle_degrees < 70 and crossing_angle_degrees > 50:
                    assign_angl = 60
        # 90 degree
                if crossing_angle_degrees < 100 and crossing_angle_degrees > 80:
                    assign_angl = 90       
        # 120 degree
                if crossing_angle_degrees < 130 and crossing_angle_degrees > 110:
                    assign_angl = 120
        # 150 degree
                if crossing_angle_degrees < 160 and crossing_angle_degrees > 140:
                    assign_angl = 150

                deviation = calculate_deviation_ang(assign_angl, crossing_angle_degrees)

                deviation_part1.append(deviation)

            groupinit1x.clear()
            groupinit1y.clear()
            groupinit2x.clear()
            groupinit2y.clear()

            groupfinal1x.clear()
            groupfinal1y.clear()
            groupfinal2x.clear()
            groupfinal2y.clear()
        
        atemp = list(deviation_part1)  # Create a copy of the list
        ext_deviat.append(atemp)
        deviation_part1.clear()

    return ext_deviat

def calculate_deviat_angl(all_csv):
    n = 1
    i = 0

    assign_velocity = []
    time = []
    file_velocity = []
    agenr_file_vel = []
    file_velocity.clear()
    atemp = []
    temp_agent_x_end = []
    temp_agent_y_end = []

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
        for n in range(1,int(temp)):
            #    Assign every value for x and y in every column for every row from one csv file
            assign_velocity.clear()
            temp_agent_x = all_csv[i].iloc[:, 2*n+1].to_numpy()    
            temp_agent_y = all_csv[i].iloc[:, 2*(n+1)].to_numpy()  

            temp_agent_x_end = all_csv[i].iloc[:, 2*n+1].to_numpy()
            temp_agent_y_end = all_csv[i].iloc[:, 2*(n+1)].to_numpy()
            #print(temp_agent_x, temp_agent_x_end) 
            #angle_deviat_overall = np.arctan2(temp_agent_y_end[0] - temp_agent_y[-1], temp_agent_x_end[0] - temp_agent_x[-1]) * 180 / np.pi            
            angle_deviat_overall = math.atan2(temp_agent_y_end[0] - temp_agent_y[-1],temp_agent_x_end[0] - temp_agent_x[-1]) * 180 / np.pi            

            #print(angle_deviat_overall)
            # Deviation for one agent at every step
            for tim in time: # dla kazdego czasu odejmuje te wartosci 
                #angle_deviat = np.arctan2(temp_agent_y_end - temp_agent_y[tim-1], temp_agent_x_end - temp_agent_x[tim-1]) * 180 / np.pi               
                angle_deviat = math.atan2(temp_agent_y_end[tim] - temp_agent_y[tim-1],temp_agent_x_end[tim] - temp_agent_x[tim-1]) * 180 / np.pi               
                #angle_deviat = np.arctan2(temp_agent_y_end[tim] - temp_agent_y[tim-1], temp_agent_x_end[tim] - temp_agent_x[tim-1]) * 180 / np.pi               
                deviation_final = angle_deviat-angle_deviat_overall + 180
                # na poczatku od pierwszego do koncowego, i pozniej odejmuje od wyznaczonego kata deviacje 
                assign_velocity.append(deviation_final)

            agenr_file_vel.extend(assign_velocity)

        atemp = list(agenr_file_vel)  # Create a copy of the list
        file_velocity.append(atemp)

    ret_val = list(file_velocity)#np.column_stack((file_names ,file_velocity))

    return ret_val

# def calculate_angle_deviat_right_one(all_csv, file_names):
    
#     n = 1 
#     n2 = 2
#     k_means_temp_init = []
#     k_means_temp_final = []
#     collect_angle_degree = []

#     groupinit1x = []
#     groupinit1y = []
#     groupinit2x = []
#     groupinit2y = []

#     groupfinal1x = []
#     groupfinal1y = []
#     groupfinal2x = []
#     groupfinal2y = []

#     deviation_part1 = []
#     ext_deviat = []
#     assign_angl = 0

#     for i in range(len(all_csv)):

#         #   Initial values
#         #   In this data set we have two groups in one variable!


#         if crossing_angle_degrees < 10:
#             assign_angl = 0
#         # 30 degree
#         if crossing_angle_degrees < 40 and crossing_angle_degrees > 20:
#             assign_angl = 30
#         # 60 degree
#         if crossing_angle_degrees < 70 and crossing_angle_degrees > 50:
#             assign_angl = 60
#         # 90 degree
#         if crossing_angle_degrees < 100 and crossing_angle_degrees > 80:
#             assign_angl = 90       
#         # 120 degree
#         if crossing_angle_degrees < 130 and crossing_angle_degrees > 110:
#             assign_angl = 120
#         # 150 degree
#         if crossing_angle_degrees < 160 and crossing_angle_degrees > 140:
#             assign_angl = 150

#         collect_angle_degree.clear()

#         #   Every agent:

#         for l1 in range(1,all_csv[i].shape[1]):

#             temp1x = all_csv[i].iloc[l1, n::(2*n)].to_numpy()    
#             temp1y = all_csv[i].iloc[l1, n2::(2*n)].to_numpy()

#         #   End value

#             temp2x = all_csv[i].iloc[-1, n::(2*n)].to_numpy()
#             temp2y = all_csv[i].iloc[-1, n::(2*n)].to_numpy()

#             group1_initial_barycenter = np.array([temp1x,temp1y])
#             group1_final_barycenter = np.array([temp2x,temp2y])


#             angle_deviat = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi


 
#             deviation = calculate_deviation_ang(assign_angl, crossing_angle_degrees)

#             deviation_part1.append(deviation)

#             groupinit1x.clear()
#             groupinit1y.clear()
#             groupinit2x.clear()
#             groupinit2y.clear()

#             groupfinal1x.clear()
#             groupfinal1y.clear()
#             groupfinal2x.clear()
#             groupfinal2y.clear()
        
#         atemp = list(deviation_part1)  # Create a copy of the list
#         ext_deviat.append(atemp)
#         deviation_part1.clear()

#     return ext_deviat

def get_segragated_files():

    # Define theoretical value on calculated data set

    # CREATE A LOOP WHICH IS CHECKIN THOSE VALUE AND DEFINE THEM INTO THEORY VALUE:

# Open the file in read mode
    with open('nameFile_0.txt', 'r') as file:
    # Read all lines from the file
        lines = file.readlines()

# Remove trailing newline characters and create a list
    values = [line.strip() for line in lines]

# Open the file in read mode
    with open('nameFile_30.txt', 'r') as file:
    # Read all lines from the file
        lines = file.readlines()

# Remove trailing newline characters and create a list
    values1 = [line.strip() for line in lines]

# Open the file in read mode
    with open('nameFile_60.txt', 'r') as file:
    # Read all lines from the file
        lines = file.readlines()

# Remove trailing newline characters and create a list
    values2 = [line.strip() for line in lines]

# Open the file in read mode
    with open('nameFile_90.txt', 'r') as file:
    # Read all lines from the file
        lines = file.readlines()

# Remove trailing newline characters and create a list
    values3 = [line.strip() for line in lines]

# Open the file in read mode
    with open('nameFile_120.txt', 'r') as file:
    # Read all lines from the file
        lines = file.readlines()

# Remove trailing newline characters and create a list
    values4 = [line.strip() for line in lines]

# Open the file in read mode
    with open('nameFile_150.txt', 'r') as file:
    # Read all lines from the file
        lines = file.readlines()

    values5 = [line.strip() for line in lines]

    return values, values1, values2, values3, values4, values5

def plot_histogram(comb_0, comb_30, comb_60, comb_90, comb_120, comb_150, velocity_arr, file_names):

    pos_0 = []
    pos_30 = []
    pos_60 = []
    pos_90 = []
    pos_120 = []
    pos_150 = []


# Open the file in read mode
    with open('deviat_0.txt', 'r') as file:
    # Read all lines from the file
        lines = file.readlines()
# Remove trailing newline characters and create a list
    deviat = [line.strip() for line in lines]

# Open the file in read mode
    with open('deviat_30.txt', 'r') as file:
    # Read all lines from the file
        lines = file.readlines()

# Remove trailing newline characters and create a list
    deviat1 = [line.strip() for line in lines]

# Open the file in read mode
    with open('deviat_60.txt', 'r') as file:
    # Read all lines from the file
        lines = file.readlines()

# Remove trailing newline characters and create a list
    deviat2 = [line.strip() for line in lines]

# Open the file in read mode
    with open('deviat_90.txt', 'r') as file:
    # Read all lines from the file
        lines = file.readlines()

# Remove trailing newline characters and create a list
    deviat3 = [line.strip() for line in lines]

# Open the file in read mode
    with open('deviat_120.txt', 'r') as file:
    # Read all lines from the file
        lines = file.readlines()

# Remove trailing newline characters and create a list
    deviat4 = [line.strip() for line in lines]

# Open the file in read mode
    with open('deviat_150.txt', 'r') as file:
    # Read all lines from the file
        lines = file.readlines()

# Remove trailing newline characters and create a list
    deviat5 = [line.strip() for line in lines]

    #print(velocity_arr)
    for i in range(len(comb_0)):
        for l in range(len(file_names)):
            if file_names[l] == comb_0[i]:
                for k in range(len(velocity_arr[l])):
                    pos_0.append(velocity_arr[l][k])
                    #print(velocity_arr[l][k])
    for i in range(len(comb_30)):
        for l in range(len(file_names)):
            if file_names[l] == comb_30[i]:
                for k in range(len(velocity_arr[l])):
                    pos_30.append(velocity_arr[l][k])
    for i in range(len(comb_60)):
        for l in range(len(file_names)):
            if file_names[l] == comb_60[i]:
                for k in range(len(velocity_arr[l])):
                    pos_60.append(velocity_arr[l][k])
    for i in range(len(comb_90)):
        for l in range(len(file_names)):
            if file_names[l] == comb_90[i]:
                for k in range(len(velocity_arr[l])):
                    pos_90.append(velocity_arr[l][k])
    for i in range(len(comb_120)):
        for l in range(len(file_names)):
            if file_names[l] == comb_120[i]:
                for k in range(len(velocity_arr[l])):
                    pos_120.append(velocity_arr[l][k])
    for i in range(len(comb_150)):
        for l in range(len(file_names)):
            if file_names[l] == comb_150[i]:
                for k in range(len(velocity_arr[l])):
                    pos_150.append(velocity_arr[l][k])

    #print(pos_0[100], pos_30[100], pos_60[100], pos_90[100], pos_120[100], pos_150[100])

    # Open the file in write mode
    with open('deviat_0_full.txt', 'w') as file:
    # Write each element of the list on a new line
        for item in pos_0:
            file.write(str(item) + '\n')

    # Open the file in write mode
    with open('deviat_30_full.txt', 'w') as file:
    # Write each element of the list on a new line
        for item in pos_30:
            file.write(str(item) + '\n')

    # Open the file in write mode
    with open('deviat_60_full.txt', 'w') as file:
    # Write each element of the list on a new line
        for item in pos_60:
            file.write(str(item) + '\n')

    # Open the file in write mode
    with open('deviat_90_full.txt', 'w') as file:
    # Write each element of the list on a new line
        for item in pos_90:
            file.write(str(item) + '\n')

    # Open the file in write mode
    with open('deviat_120_full.txt', 'w') as file:
    # Write each element of the list on a new line
        for item in pos_120:
            file.write(str(item) + '\n')

    # Open the file in write mode
    with open('deviat_150_full.txt', 'w') as file:
    # Write each element of the list on a new line
        for item in pos_150:
            file.write(str(item) + '\n')

# main

degree_0 = []
degree_30 = []
degree_60 = []
degree_90 = []
degree_120 = []
degree_150 = []

all_csv, file_names = read_all_csv()
deviations_angles = calculate_deviat_angl(all_csv)
#print(deviations_angles[0])
degree_0, degree_30, degree_60, degree_90, degree_120, degree_150 = get_segragated_files()
plot_histogram(degree_0, degree_30, degree_60, degree_90, degree_120, degree_150, deviations_angles, file_names)

#print(deviations_angles)

