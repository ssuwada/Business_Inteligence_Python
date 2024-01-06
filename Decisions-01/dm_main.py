#   ----------- CLASS EXERCISES -----------
#           Decision making problem
#           ---------------------
#              SEBASTIAN SUWADA 
#           ---------------------
#
# ------------- Import files --------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import time
import math

# ---- Defined files ----
from dm_simulation import *

ts = time()
# p = np.linspace(0, 1, num=100)
p = 0, 0.5
Sim1 = Simulation(50,0.5,p,100000,5)
Sim1.Simulation_full()
# print(vector)
time_taken = time() - ts
print(f'Done in {time_taken} seconds')