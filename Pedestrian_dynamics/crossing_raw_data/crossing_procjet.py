
#   ---------PROJECT CROSSING ANGLE-----------
#                Report task
#             till 20th of May
#           ---------------------
#              SEBASTIAN SUWADA 
#           ---------------------
#           Angle calculation part

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
                          
    print(file_names)

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
        silhouette_avg = silhouette_score(combined_arr_init1xy, k_means_temp_init[i])
        # print clusters labels  

        #print(silhouette_avg)
        #print(k_means_temp_init[i])

        #   PERFORM CLUSTERING FIRST FINAL GROUPS

        # cluster each set of data using K-means
        k_means_temp_final.append(k_means(combined_arr_init2xy))
        # chceck score of labels for clusters:
        silhouette_avg = silhouette_score(combined_arr_init2xy, k_means_temp_final[i])
        # print clusters labels  

        #print(silhouette_avg)
        #print(k_means_temp_final[i])

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

        # Create a figure with two subplots
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # ax1.scatter(groupinit1x,groupinit1y, c = 'black')
        # ax1.scatter(groupinit2x,groupinit2y, c = 'red')

        # ax1.scatter(groupfinal1x,groupfinal1y, c = 'blue')
        # ax1.scatter(groupfinal2x,groupfinal2y, c = 'gray')

        #   Create 4 barycenters each for every group (initial1, initial2, final1, final2)
        #   Value of each variable (output) are coordinates of barycenter

        group1_initial_barycenter = np.array([mean(groupinit1x), mean(groupinit1y)])
        group1_final_barycenter = np.array([mean(groupfinal1x), mean(groupfinal1y)])
        group2_initial_barycenter = np.array([mean(groupinit2x), mean(groupinit2y)])
        group2_final_barycenter = np.array([mean(groupfinal2x), mean(groupfinal2y)])

        # ax2.scatter(group1_initial_barycenter[0],group1_initial_barycenter[1],marker='D', color='orange', s=100)
        # ax2.scatter(group1_final_barycenter[0],group1_final_barycenter[1],marker='D', color='orange', s=100)
        # ax2.scatter(group2_initial_barycenter[0],group2_initial_barycenter[1],marker='p', color='orange', s=100)
        # ax2.scatter(group2_final_barycenter[0],group2_final_barycenter[1],marker='p', color='orange', s=100)

        # ax1.scatter(group1_initial_barycenter[0],group1_initial_barycenter[1],marker='D', color='orange', s=100)
        # ax1.scatter(group1_final_barycenter[0],group1_final_barycenter[1],marker='D', color='orange', s=100)
        # ax1.scatter(group2_initial_barycenter[0],group2_initial_barycenter[1],marker='p', color='orange', s=100)
        # ax1.scatter(group2_final_barycenter[0],group2_final_barycenter[1],marker='p', color='orange', s=100)


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

        # x = [group1_initial_barycenter[0], group1_final_barycenter[0]]
        # y = [group1_initial_barycenter[1], group1_final_barycenter[1]]

        # x1 = [group2_initial_barycenter[0], group2_final_barycenter[0]]
        # y1 = [group2_initial_barycenter[1], group2_final_barycenter[1]]

        # ax2.plot(x,y, color='blue', label='Group 1')
        # ax2.plot(x1,y1, color='green', label='Group 2')


        # Set y-axis and x-axis for scientific notation
        # formatter = ticker.ScalarFormatter(useMathText=True)
        # formatter.set_powerlimits((-3, 3))
        # ax1.xaxis.set_major_formatter(formatter)
        # ax1.yaxis.set_major_formatter(formatter)
        # ax2.xaxis.set_major_formatter(formatter)
        # ax2.yaxis.set_major_formatter(formatter)
        # ax1.set_title('Scatteer plot of groups - '+file_names[i])
        # ax2.set_title('Intersection of vectors - '+file_names[i])
        # # plt.xlabel('X-axis [mm]')
        # # plt.ylabel('Y-axis [mm]')
        # plt.legend()


        # plt.show()

        groupinit1x.clear()
        groupinit1y.clear()
        groupinit2x.clear()
        groupinit2y.clear()

        groupfinal1x.clear()
        groupfinal1y.clear()
        groupfinal2x.clear()
        groupfinal2y.clear()
    
    return collect_angle_degree

def calculate_deviation_ang(assign_ang,comput_ang):
    deviation = assign_ang - comput_ang
    return deviation

def define_theo_val(file_names, cross_ang):
    # Combine two data sets into 2D array
    combined_arr = np.column_stack((file_names, cross_ang))

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
    deviat_0 = []
    deviat_30 = []
    deviat_60 = []
    deviat_90 = []
    deviat_120 = []
    deviat_150 = []
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
            deviat_0.append(calculate_deviation_ang(cross_assign,cross_ang[i]))

        # 30 degree
        if cross_ang[i] < 40 and cross_ang[i] > 20:
            degree_30_ang.append(cross_ang[i])
            degree_30_name.append(file_names[i])
            cross_assign = 30
            deviat_30.append(calculate_deviation_ang(cross_assign,cross_ang[i]))

        # 60 degree
        if cross_ang[i] < 70 and cross_ang[i] > 50:
            degree_60_ang.append(cross_ang[i])
            degree_60_name.append(file_names[i])
            cross_assign = 60
            deviat_60.append(calculate_deviation_ang(cross_assign,cross_ang[i]))

        # 90 degree
        if cross_ang[i] < 100 and cross_ang[i] > 80:
            degree_90_ang.append(cross_ang[i])
            degree_90_name.append(file_names[i])
            cross_assign = 90
            deviat_90.append(calculate_deviation_ang(cross_assign,cross_ang[i]))

        # 120 degree
        if cross_ang[i] < 130 and cross_ang[i] > 110:
            degree_120_ang.append(cross_ang[i])
            degree_120_name.append(file_names[i])
            cross_assign = 120
            deviat_120.append(calculate_deviation_ang(cross_assign,cross_ang[i]))

        # 150 degree
        if cross_ang[i] < 160 and cross_ang[i] > 140:
            degree_150_ang.append(cross_ang[i])
            degree_150_name.append(file_names[i])
            cross_assign = 150
            deviat_150.append(calculate_deviation_ang(cross_assign,cross_ang[i]))


        # 180 degree - same as 0 degree
    comb_0 = np.column_stack((degree_0_name, degree_0_ang, deviat_0))
    comb_30 = np.column_stack((degree_30_name, degree_30_ang, deviat_30))
    comb_60 = np.column_stack((degree_60_name, degree_60_ang, deviat_60))
    comb_90 = np.column_stack((degree_90_name, degree_90_ang, deviat_90))
    comb_120 = np.column_stack((degree_120_name, degree_120_ang, deviat_120))
    comb_150 = np.column_stack((degree_150_name, degree_150_ang, deviat_150))

    # Create DataFrame
    df = pd.DataFrame(combined_arr, columns=['File_Name', 'Computed_Angle'])
    # Write this data frame into excel file as output
    df.to_excel('angles.xlsx', index=False)


    # Create DataFrame from each of the 2D arrays
    df1 = pd.DataFrame(comb_0, columns=['File_Name', 'Computed_Angle', 'Deviation angle'])
    df2 = pd.DataFrame(comb_30, columns=['File_Name', 'Computed_Angle', 'Deviation angle'])
    df3 = pd.DataFrame(comb_60, columns=['File_Name', 'Computed_Angle', 'Deviation angle'])
    df4 = pd.DataFrame(comb_90, columns=['File_Name', 'Computed_Angle', 'Deviation angle'])
    df5 = pd.DataFrame(comb_120, columns=['File_Name', 'Computed_Angle', 'Deviation angle'])
    df6 = pd.DataFrame(comb_150, columns=['File_Name', 'Computed_Angle', 'Deviation angle'])
    
    # Write information about Data - seperate for every theoretical value of angle:
    # create a writer object for the Excel file

    with pd.ExcelWriter("angles_devided.xlsx") as writer:
    # write each DataFrame to the Excel file with separation of one column between them
        df1.to_excel(writer, sheet_name='Sheet1', index=False)
        df2.to_excel(writer, sheet_name='Sheet1', index=False, startcol=3)
        df3.to_excel(writer, sheet_name='Sheet1', index=False, startcol=6)
        df4.to_excel(writer, sheet_name='Sheet1', index=False, startcol=9)
        df5.to_excel(writer, sheet_name='Sheet1', index=False, startcol=12)
        df6.to_excel(writer, sheet_name='Sheet1', index=False, startcol=15)

   # Open the file in write mode
    with open('deviat_0.txt', 'w') as file:
    # Write each element of the list on a new line
        for item in deviat_0:
            file.write(str(item) + '\n')

    # Open the file in write mode
    with open('deviat_30.txt', 'w') as file:
    # Write each element of the list on a new line
        for item in deviat_30:
            file.write(str(item) + '\n')

    # Open the file in write mode
    with open('deviat_60.txt', 'w') as file:
    # Write each element of the list on a new line
        for item in deviat_60:
            file.write(str(item) + '\n')

    # Open the file in write mode
    with open('deviat_90.txt', 'w') as file:
    # Write each element of the list on a new line
        for item in deviat_90:
            file.write(str(item) + '\n')

    # Open the file in write mode
    with open('deviat_120.txt', 'w') as file:
    # Write each element of the list on a new line
        for item in deviat_120:
            file.write(str(item) + '\n')

    # Open the file in write mode
    with open('deviat_150.txt', 'w') as file:
    # Write each element of the list on a new line
        for item in deviat_150:
            file.write(str(item) + '\n')

# Open the file in write mode
    with open('nameFile_0.txt', 'w') as file:
    # Write each element of the list on a new line
        for item in degree_0_name:
            file.write(str(item) + '\n')

    # Open the file in write mode
    with open('nameFile_30.txt', 'w') as file:
    # Write each element of the list on a new line
        for item in degree_30_name:
            file.write(str(item) + '\n')

    # Open the file in write mode
    with open('nameFile_60.txt', 'w') as file:
    # Write each element of the list on a new line
        for item in degree_60_name:
            file.write(str(item) + '\n')

    # Open the file in write mode
    with open('nameFile_90.txt', 'w') as file:
    # Write each element of the list on a new line
        for item in degree_90_name:
            file.write(str(item) + '\n')

    # Open the file in write mode
    with open('nameFile_120.txt', 'w') as file:
    # Write each element of the list on a new line
        for item in degree_120_name:
            file.write(str(item) + '\n')

    # Open the file in write mode
    with open('nameFile_150.txt', 'w') as file:
    # Write each element of the list on a new line
        for item in degree_150_name:
            file.write(str(item) + '\n')


# --------- Main ----------

all_csv = []
file_names = []

all_csv, file_names = read_all_csv()

n = 1
n2 = 2

crossing_angle_degree = calculate_angle(all_csv,file_names)
define_theo_val(file_names, crossing_angle_degree)

# ----------- BRUDNOPIS -----------


# print(all_csv[0].iloc[1, n::(2*n)])
# print(all_csv[0].iloc[1, n2::(2*n)])


# Initial positions for every file stored in all_csv, we see first row of every file for every agent 

#for i in range(len(all_csv[0].iloc[1, n::(2*n)])):

    #   Initial values

  #  temp1x = all_csv[i].iloc[1, n::(2*n)].to_numpy()
 #   temp1y = all_csv[i].iloc[1, n2::(2*n)].to_numpy()

    #   End values

   # temp2x = all_csv[i].iloc[-1, n::(2*n)].to_numpy()
 #   temp2y = all_csv[i].iloc[-1, n2::(2*n)].to_numpy()

    #plt.scatter(all_csv[i].iloc[1, n::(2*n)],all_csv[i].iloc[1, n2::(2*n)]) #initial values
    #plt.scatter(all_csv[i].iloc[-1, n::(2*n)],all_csv[i].iloc[-1, n2::(2*n)]) #end values

    #plt.scatter(temp2x,temp2y)
    #plt.scatter(temp1x,temp1y)

    #plt.show()

    # K = 2
    # rnd = random.randrange(len(temp1))
    # rnd2 = random.randrange(len(temp1))


    # CentroidsX = (temp1[rnd],temp1[rnd2])
    # CentroidsY = (temp2[rnd],temp2[rnd2])

    # print(CentroidsX,CentroidsY)

    # plt.scatter(all_csv[s].iloc[1, n::(2*n)],all_csv[s].iloc[1, n2::(2*n)])

    # plt.scatter(CentroidsX,CentroidsY,c='red')

    #plt.scatter(all_csv[s].iloc[0,1],all_csv[s].iloc[0,1])



#plt.scatter(all_csv[i].iloc[1, n::(2*n)],all_csv[i].iloc[1, n2::(2*n)]) #initial values
#plt.scatter(all_csv[i].iloc[-1, n::(2*n)],all_csv[i].iloc[-1, n2::(2*n)]) #end values
