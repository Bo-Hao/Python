
import numpy as np 
from math import * 
import matplotlib.pyplot as plt

def f(a, e):
    # a: angle e: elevation
    a = a/180*np.pi
    e = e/180*np.pi
    new_angle = acos(np.sin(e)**2 + np.cos(a) * np.cos(e)**2)
    return new_angle/np.pi*180


bafs = [[1, 9], [1, 37], [1, 59], [2, 18], [2, 34], [2, 48]]

new_bafs = [(i[0]+i[1]/60)/180*np.pi for i in bafs]

color = ['red', 'blue', 'green', 'black', 'gray', 'pink']

for baf in range(len(bafs)):
    for j in np.arange(1, 89, 0.5):
        plt.scatter(j, f(new_bafs[baf], j), c = color[baf], s= 0.5)
        plt.scatter(j, -f(new_bafs[baf], j), c = color[baf], s= 0.5)
        plt.scatter(-j, f(new_bafs[baf], j), c = color[baf], s= 0.5)
        plt.scatter(-j, -f(new_bafs[baf], j), c = color[baf], s= 0.5)
plt.show()


