import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import time
import math
import csv



p = np.linspace(0, 1, num=1000)
p2 = np.linspace(0, 1, num=100)
p_final = []
i = 0
for i in range(0, 101, 1):
    p = round(i / 100, 2)
    p_final.append(p)
print(p_final)
E = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3e-05, 0.0001, 0.00015, 0.00036, 0.00095, 0.00202, 0.00489, 0.01253, 0.03179, 0.07923, 0.18706, 0.3741, 0.49664, 0.61637, 0.81442, 0.92097, 0.96771, 0.9872, 0.99471, 0.99772, 0.9992, 0.99965, 0.99987, 0.99991, 0.99995, 0.99999, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
E2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3e-05, 0.0001, 0.00015, 0.00036, 0.00095, 0.00202, 0.00489, 0.01253, 0.03179, 0.07923, 0.18706, 0.3741, 0.61637, 0.81442, 0.92097, 0.96771, 0.9872, 0.99471, 0.99772, 0.9992, 0.99965, 0.99987, 0.99991, 0.99995, 0.99999, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


plt.plot(p2,E2)
plt.grid(True)
plt.xlabel('Probability')
plt.ylabel('Exit probability')
plt.title('Plot of exit probability E as a function of p')
plt.show()

temp1 = []
temp2 = []

for i in range(len(p2)):
    if E[i] != 1 or E[i] != 0:
        temp1.append(E[i])
        temp2.append(p_final[i])

# filename = 'list_exitP.csv'

# # Open the file in write mode and create a CSV writer object
# with open(filename, 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)

#     # Write the lists to the CSV file as separate rows
#     writer.writerow(temp1)
#     writer.writerow(temp2)

# print("List saved to CSV file successfully.")


# 50 agents can not be convinced, and after everyone can be go for of plus, write name of the function and describe
# all things from conversation using good words(not like in mail), give value for 50% and for reaching 1 this value also
# can start from 0.99 so the first value in my case will be 0.56
# its called step function
# and universality class, 
# i can find something that is meaningful for additional task and make description about it also