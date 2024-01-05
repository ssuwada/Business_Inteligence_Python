#   -----------  PROJCET CLASS  -----------
#              Simulation class
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
import random

class Simulation():

    def __init__(self, vector_size, x: float, p: float, iterations: int, MCs: int):
        self.vector_size = vector_size             # Size of vector 1D
        self.x = x                                 # Initial percentage of agreement of all agents
        self.p = p                                 # Probability to change opinion
        self.iterations = iterations                                
        self.MCs = MCs
        self.vector = self.create_vector(self.vector_size,self.x)

    def create_vector(self, vector_size, x):
        
        agents_positive = int(vector_size*x)
        agents_negative = vector_size - agents_positive

        vector = []

        for i in range(agents_positive):
            vector.append(1)
        for i in range(agents_negative):
            vector.append(-1)

        np.random.shuffle(vector)

        return vector

    def d_Energy(self, pra, actu, next_):
        add = (next_+pra)
        dE = 2*actu*add
        return dE

    def energy_calculate_decision(self,p):
        #ts = time()
        a = [random.randint(0, 49) for _ in range(49)]
        random_list = [np.random.random() for _ in range(len(self.vector)-1)]

        ## OPTIMIZED FOR LOOP FOR 
        #   Utilize modular arithmetic for boundary conditions:
        #   Instead of using separate conditional statements to 
        #   handle boundary conditions, you can use modular arithmetic 
        #   to wrap the indices around. This avoids the need for 
        #   multiple if-else statements.

        #print(self.vector.count(1))

        for i in range(len(self.vector) - 1):
            pre = self.vector[(a[i] - 1) % 50]
            actual = self.vector[a[i]]
            next_ = self.vector[(a[i] + 1) % 50]
            
            dE = self.d_Energy(pre, actual, next_)

            if dE == 0:
                if self.vector[a[i]] == -1 and random_list[i] < p and dE == 0:
                        self.vector[a[i]] *= -1
                        break
                if self.vector[a[i]] == 1 and random_list[i] < 1-p and dE == 0:
                        self.vector[a[i]] *= -1
                        break
            elif dE < 0:
                self.vector[a[i]] *= -1

    def oneMCS(self,p):
        #p = np.linspace(0, 1, num=100)
        c1 = 0
        ticks = 0

        ts = time()

        while ticks != 3000:
             self.energy_calculate_decision(p)
             c1 = self.vector.count(1)
             ticks += 1
             if c1 == 0 or c1 == len(self.vector):
                  break
       
        return c1
             
    def MonteCarloSim(self, iterations,p):
        ite_list = []
        # ts = time()
        for ite in range(iterations):
            self.vector = self.create_vector(self.vector_size,self.x)
            ite_list.append(self.oneMCS(p))
            #print(f'{ite+1} out of {iterations} finished..')
        
        # print(ite_list)
        # time_taken = time() - ts
        # print(f'Taken time: '+str(time_taken))

        return ite_list

    def Simulation_full(self):
        temp_list = []
        fraction = []
        for p_ in self.p:
            ts = time()
            print(f'Calculating for probability: {p_}')
            temp_list = self.MonteCarloSim(self.iterations,p_)
            coun = temp_list.count(len(self.vector))
            fraction.append(coun/self.iterations)
            time_taken = time() - ts
            print(f'Taken time: '+str(time_taken))    

        print(fraction)

        # Load data into file:
        filename = "Probability_P2_"+str(self.iterations)+".txt"
        with open(filename, "w") as f:
            for item in fraction:
                f.write(str(item) + '\n')

# Czyli robimy graph prawdopodobienstwa -> ze wszystkie agenty sa na tak lub nie. 
#             random_list = [np.random.random() for _ in range(len(vector))]
#         [a.append(i+1) for i in range(-1,len(vector)-1)]
