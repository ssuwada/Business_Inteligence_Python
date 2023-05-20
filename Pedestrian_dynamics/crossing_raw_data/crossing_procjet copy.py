
#   ---------PROJECT CROSSING ANGLE-----------
#                Report task
#             till 20th of May
#           ---------------------
#              SEBASTIAN SUWADA 
#           ---------------------

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import glob
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def read_all_csv():

# Define the path to the directory containing the CSV files
    path = '/Users/sebastiansuwada/Desktop/Python_Practice/Python_Project_BI/Pedestrian_crossing/crossing_flows_data/*.csv'

# Use the glob function to create a list of file names that match a pattern
# For example, all files with the extension .csv in the specified directory
    all_files = glob.glob(path)

# Create an empty list to store the dataframes
    list_of_dataframes = []

# Iterate over the list of file names and read each file into a pandas dataframe
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        list_of_dataframes.append(df)

    return list_of_dataframes

def k_means(all_csv):
    n = 1 
    n2 = 2

    k_means_temp = []

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

        # Normalize the feature data
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(combined_arr_init1xy)

        # cluster each set of data using K-means
        kmeans1 = KMeans(n_clusters=2)
        kmeans1.fit(normalized_features)
        predicted = kmeans1.predict(normalized_features)

        #kmeans2 = KMeans(n_clusters=2).fit(combined_arr_init2xy)

        k_means_temp.append(predicted)

    print(k_means_temp[0])
    # dodac do petli zeby combined bylo dla kazdego, wtedy moze mi sie zgadzac kmeans!
    groupinit1x = []
    groupinit1y = []

    groupinit2x = []
    groupinit2y = []

    print(len(k_means_temp))
    l = 0

    for j in range(len(k_means_temp)):
        #print(k_means_temp[j])

        for l in range(len(k_means_temp[j])-2):
            #print(k_means_temp[j][l])

            if k_means_temp[j][l] == 0:
                print(combined_arr_init1xy[l][0])
                groupinit1x.append(combined_arr_init1xy[l][0])
                groupinit1y.append(combined_arr_init1xy[l][1])
            if k_means_temp[j][l] == 1:
                groupinit2x.append(combined_arr_init1xy[l][0])
                groupinit2y.append(combined_arr_init1xy[l][1])

        plt.scatter(groupinit1x,groupinit1y, c = 'black')
        plt.scatter(groupinit2x,groupinit2y, c = 'red')

        groupinit1x.clear()
        groupinit1y.clear()
        groupinit2x.clear()
        groupinit2y.clear()

        #plt.scatter(groupinit2x, groupinit2y, c = 'blue')

        plt.show()

        #kmeans2 = KMeans(n_clusters=2, random_state=0).fit(temp1y)

        #plt.scatter(temp2x,temp2y)
        #plt.scatter(temp1x,temp1y)

        #plt.show()


# --------- Main ----------

all_csv = []

all_csv = read_all_csv()

n = 1
n2 = 2

k_means(all_csv)

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


# ----------- BRUDNOPIS -----------

#plt.scatter(all_csv[i].iloc[1, n::(2*n)],all_csv[i].iloc[1, n2::(2*n)]) #initial values
#plt.scatter(all_csv[i].iloc[-1, n::(2*n)],all_csv[i].iloc[-1, n2::(2*n)]) #end values
