#------------------  IMPORT   -----------------

import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import norm

#-------------------------------------------------------
#------------------  Sebastian Suwada  -----------------
#-------------------------------------------------------

#------------------  LIST OF EXERCISES   -----------------


#   Ex. 1 Write a script that imports numpy library (typically import numpy as np), 
#   then produces 1000 uniformly distributed numbers from the interval [0, 1) (use np.random.random(...) function)

#   Ex. 2 Write a script that produces a list of 1000 random variables with normal distribution (np.random.normal(...), see the documentation for details) with:
#   a) mean 0 and variance 1
#   b) mean 2.2 and variance 3.8

#   Ex. 3 Import the matplotlib library (import matplotlib.pyplot as plt) Using the data generated in task 2:
#   a) plot the consecutive observaions from point a) (use dots without a connecting line, plt.plot(..., ‘.’)).
#   b) create a histogram (using densities, not counts of occurences) with 30 bins of the data generated in point b) (use plt.hist(..., bins=30))
#   c) import   package and check (in the documentation of ), how to generate a probability density function. 
#   Then, plot it on top of the histogram using red, dotted line.

#   Ex. 4 Write a program that:
#   a) produces a list of the first N prime numbers. Do not use any libraries to do that, 
#   you can reuse your own code form List 2. Make N a parameter in the code (allow the user to change it directly in the code, 
#   no need for an interactive input), and for the submission use N=114
#   b) computes a list of the consecutive proportions between the difference of the two adjacent prime numbers and the smaller of them, 
#   i.e. series of elements (p(i+1) – p(i)) / p(i), where p(k) denotes the k-th prime number
#   c) plots the series obtained in the previous point using a black line
#   d) writes the sum of the first N numbers in the title of the plot 
#   (the program should compute the sum from the list) and an information, whether this sum is a prime number itself. 
#   If yes, write which prime number it is (in the list of all prime numbers, e.g., 
#   number 7 is a 4th prime number, so if 7 was the sum, then the title would say: “The sum is 7, which is a 4th prime number” or “... which is not a prime number” if the sum is not a prime number)


#------------------  FUNC   -----------------

#   Unifomr distribution function
#       l - low band
#       m - max band
#       s - size - amount of elements

def rnd_uniform(l,m,s):
    return np.random.uniform(l,m,s)

#   Normal distribution function
#       np.random.normal(mean, standard deviation, size) 
#       np.random.normal(mean, sqrt(variance), size) -> by given variance

def rnd_normal(mu, sigma, s):
    return np.random.normal(mu,np.sqrt(sigma),s)

#   Scatter functions - Dotted function plot

def plot_distribution_dot(dist):
    x = np.linspace(0,10,1000)
    plt.scatter(x,dist, marker = '.')
    plt.show()

#   Histogram function with Bins + Density func
#   alpha = opacity of dots, lw = conection line width, marker - type of plot

def plot_histogram(dist,mean,variance):
    x = np.linspace(-4,8,100)
    plt.hist(dist, density=True, bins=30, histtype='stepfilled', color='cornflowerblue', edgecolor='black')
    plt.plot(x,norm.pdf(x, mean, np.sqrt(variance)),'r', lw=0, alpha=0.9, marker = '.')
    plt.show()

#   Prime numbers generator to - import to list

def primeNumbers(N):
    prime_list = []
    i = 0
    #for i in range(N):
    while len(prime_list) != N: 
        flag = False
        if i > 1:
            for b in range(2,i):
                if (i % b) == 0:
                    flag = True
                    break
                else:
                    flag = False
        else:
            flag = True

        if flag == False:
            prime_list.append(i)
        i = i + 1
    return prime_list

#   Consecutive proportion 

def consecutive_proportion(k,prime_list):
    cons_list = []
    i = 0
    for i in range(k):
        cons_list.append((prime_list[i+1]-prime_list[i])/prime_list[i])
    return cons_list

#   Plot consecutive proportion numbers:

def plot_cons(cons_list):
    plt.plot(cons_list, color = 'black')
    plt.show()

#   Sum of prime numbers:

def Sum_Prime(prime_list,N):
    s = 0
    for i in range(N):
        s = prime_list[i] + s
    return s

#   Return information about sum of prime numbers..

def prime_info(prime_list,s):
    flag = False
    for i in range(len(prime_list)):
        if prime_list[i] == s:
            if i == 1:
                print(f"The sum is  "+str(s)+f"  which is  "+str(i)+f"st  prime number on the list")
            elif i == 2:
                print(f"The sum is  "+str(s)+f"  which is  "+str(i)+f"nd  prime number on the list")
            elif i == 3:
                print(f"The sum is  "+str(s)+f"  which is  "+str(i)+f"rd  prime number on the list")
            else:  
                print(f"The sum is  "+str(s)+f"  which is  "+str(i)+f"th  prime number on the list")
            flag = True
            break
    if flag == False:
        print(f"The sum is  "+str(s)+f"  which is not a prime number..") 


#------------------  MAIN   -----------------


#   Ex. 1

l = 0
m = 1
s = 1000
U1 = rnd_uniform(l,m,s)


#   Ex. 2 (a)

mean1 = 0
variance = 1
size = 1000
N1 = rnd_normal(mean1,variance, size)
print(N1)

#   Ex. 2 (b)

mean2 = 2.2
variance2 = 3.8
N2 = rnd_normal(mean2,variance2,size)
print(N2)

#   Ex. 3  (a)

plot_distribution_dot(N1)

#   Ex. 3 (b) & (c)

plot_histogram(N2,mean2,variance2)

#   Ex. 4 (a) - prime numbers
#   Nth - Border for generating prime numbers

Nth = 114   
prime_list = primeNumbers(Nth)
print(prime_list)

#   Ex. 4 (b) - consecutive proportion
#   k - Range of consecutive proportion list 

k = 100
cons_list = consecutive_proportion(k,prime_list)
#print(cons_list)

#   Ex. 4 (c) - plot consecutive proportion

plot_cons(cons_list)

#   Ex. 5 (d) - Write sum of first N numbers
#   SumN - Sum of N numbers from prime list.

SumN = 4
s = Sum_Prime(prime_list,SumN)
prime_info(prime_list,s)

