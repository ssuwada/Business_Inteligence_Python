#   ---------PROJECT CROSSING ANGLE-----------
#                Report task
#             till 20th of May
#           ---------------------
#              SEBASTIAN SUWADA 
#           ---------------------
#             Histogram plot part

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

def histogram(values01,values02, values03, values04, values05,values06, a1):
    # Plot histogram
# Normalize the data to range [0, 1]

    # Plot histogram

    fig, ax = plt.subplots()

    num_bins = 500
    n, bins = np.histogram(values01, bins=num_bins, density=True)
    n = n/np.sum(n)

    ax.plot(bins[:-1],n, label='0 degree')

    n, bins = np.histogram(values02, bins=num_bins, density=True)
    n = n/np.sum(n)

    ax.plot(bins[:-1],n, label='30 degree')

    n, bins = np.histogram(values03, bins=num_bins, density=True)
    n = n/np.sum(n)

    ax.plot(bins[:-1],n, label='60 degree')

    n, bins = np.histogram(values04, bins=num_bins, density=True)
    n = n/np.sum(n)

    ax.plot(bins[:-1],n, label='90 degree')

    n, bins = np.histogram(values05, bins=num_bins, density=True)
    n = n/np.sum(n)

    ax.plot(bins[:-1],n, label='120 degree')

    n, bins = np.histogram(values06, bins=num_bins, density=True)
    n = n/np.sum(n)

    ax.plot(bins[:-1],n, label='150 degree')

    # Set labels and title
    plt.xlabel('Velocity [m/s]')
    plt.ylabel('Relative Frequency')
    plt.title('Normalised histogram of velocity')
    plt.xlim(0.025, 2.75)  # Set the x-axis limits
    plt.ylim(0, 0.0175)
    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()

def histogram_deviat(values01,values02, values03, values04, values05,values06,a):

    # Normalize the data to range [0, 1]

    # Plot histogram

    fig, ax = plt.subplots()

    num_bins = 500
    n, bins = np.histogram(values01, bins=num_bins, density=True)
    n = n/np.sum(n)

    ax.plot(bins[:-1],n, label='0 degree')

    n, bins = np.histogram(values02, bins=num_bins, density=True)
    n = n/np.sum(n)

    ax.plot(bins[:-1],n, label='30 degree')

    n, bins = np.histogram(values03, bins=num_bins, density=True)
    n = n/np.sum(n)

    ax.plot(bins[:-1],n, label='60 degree')

    n, bins = np.histogram(values04, bins=num_bins, density=True)
    n = n/np.sum(n)

    ax.plot(bins[:-1],n, label='90 degree')

    n, bins = np.histogram(values05, bins=num_bins, density=True)
    n = n/np.sum(n)

    ax.plot(bins[:-1],n, label='120 degree')

    n, bins = np.histogram(values06, bins=num_bins, density=True)
    n = n/np.sum(n)

    ax.plot(bins[:-1],n, label='150 degree')

    # Set labels and title
    plt.xlabel('Deviation angle')
    plt.ylabel('Relative Frequency')
    plt.title('Normalised histogram of deviation')
    plt.xlim(-30, 30)  # Set the x-axis limits
    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()


# Open the file in read mode
with open('pos_0.txt', 'r') as file:
    # Read all lines from the file
    lines = file.readlines()

# Remove trailing newline characters and create a list
values = [float(line.strip()) for line in lines]

# Open the file in read mode
with open('pos_30.txt', 'r') as file:
    # Read all lines from the file
    lines = file.readlines()

# Remove trailing newline characters and create a list
values1 = [float(line.strip()) for line in lines]

# Open the file in read mode
with open('pos_60.txt', 'r') as file:
    # Read all lines from the file
    lines = file.readlines()

# Remove trailing newline characters and create a list
values2 = [float(line.strip()) for line in lines]

# Open the file in read mode
with open('pos_90.txt', 'r') as file:
    # Read all lines from the file
    lines = file.readlines()

# Remove trailing newline characters and create a list
values3 = [float(line.strip()) for line in lines]

# Open the file in read mode
with open('pos_120.txt', 'r') as file:
    # Read all lines from the file
    lines = file.readlines()

# Remove trailing newline characters and create a list
values4 = [float(line.strip()) for line in lines]

# Open the file in read mode
with open('pos_150.txt', 'r') as file:
    # Read all lines from the file
    lines = file.readlines()

# Remove trailing newline characters and create a list
values5 = [float(line.strip()) for line in lines]



# Open the file in read mode
with open('deviat_0_full.txt', 'r') as file:
    # Read all lines from the file
    lines = file.readlines()

# Remove trailing newline characters and create a list
values01 = [float(line.strip()) for line in lines]

# Open the file in read mode
with open('deviat_30_full.txt', 'r') as file:
    # Read all lines from the file
    lines = file.readlines()

# Remove trailing newline characters and create a list
values02 = [float(line.strip()) for line in lines]

# Open the file in read mode
with open('deviat_60_full.txt', 'r') as file:
    # Read all lines from the file
    lines = file.readlines()

# Remove trailing newline characters and create a list
values03 = [float(line.strip()) for line in lines]

# Open the file in read mode
with open('deviat_90_full.txt', 'r') as file:
    # Read all lines from the file
    lines = file.readlines()

# Remove trailing newline characters and create a list
values04 = [float(line.strip()) for line in lines]

# Open the file in read mode
with open('deviat_120_full.txt', 'r') as file:
    # Read all lines from the file
    lines = file.readlines()

# Remove trailing newline characters and create a list
values05 = [float(line.strip()) for line in lines]

# Open the file in read mode
with open('deviat_150_full.txt', 'r') as file:
    # Read all lines from the file
    lines = file.readlines()

# Remove trailing newline characters and create a list
values06 = [float(line.strip()) for line in lines]

a1 = 0
a2 = 30
a3 = 60
a4 = 90
a5 = 120
a6 = 150

histogram(values, values1,values2, values3, values4, values5, a1)
# histogram(values1,a2)
# histogram(values2,a3)
# histogram(values3,a4)
# histogram(values4,a5)
# histogram(values5,a6)

histogram_deviat(values01,values02, values03, values04, values05,values06, a1)
# histogram_deviat(values02,a2)
# histogram_deviat(values03,a3)
# histogram_deviat(values04,a4)
# histogram_deviat(values05,a5)
# histogram_deviat(values06,a6)
