import scipy.linalg as sl 
import numpy as np 


p = np.pi 

A = [[p**3/3, p**4/4, p**5/5], [p, p**2/2, p**3/3], [p**2/2, p**3/3, p**4/4]]
b = [ p**2-4, 2, p]
'''

P, L, U = sl.lu(A)

L_in = np.linalg.inv(L)

Lb = np.dot(L_in, b)

U_in = np.linalg.inv(U)

e1 = [-3/2/p, -p/6, 1]
e2 = [p**2-4, 2, p]

print(np.dot(e1, e2))
print(np.dot(L_in, b))
ULb = np.dot(U_in, Lb)
#print(ULb)


L = [[1, 0, 0], [3/p**2, 1, 0], [3/2/p, p/6, 1]]
U = [[p**3/3, p**4/4, p**5/5], [0, -p**2/4, -4/15*p**3], [0, 0, -p**4/180]]
A = np.array(A)
L_in = np.linalg.inv(L)
#print(np.dot(L_in, b))
print(L_in)

L2 = [[1, 0, 0], [-3/p**2, 1, 0], [-1/p, -p/6, 1]]
print(np.dot(L2, b))

print(4/p - p/3)
'''

def f(x):
    
    return -1/2+ (3)/np.pi*np.cos(x)+ (-1)/np.pi*np.sin(x) + (3)/np.pi*np.sin(2*x)
def step(x):
    if x >np.pi/2 and np.pi>=x:
        y = -2
    else:
        y = 1
    return y
    
x = np.linspace(0, np.pi/2, 20)
import matplotlib.pyplot as plt 


for i in np.arange(0, np.pi, 0.01):
    plt.scatter(i, f(i), c = 'blue')
    plt.scatter(i, step(i), c = 'red')
plt.show()

