import numpy as np
import matplotlib.pyplot as plt 


def fcn(y, f, la):
    x = (f - la*y**2)/4
    return x ** 0.5
lam = [4, 2, 1, 0, -1, -2, -4]
f_list = [0, 1, 2, 4]

y = np.linspace(-5, 5, 10000)

for i in range(len(lam)):
    f = f_list[3]
    plt.subplot(3, 3, i+1)
    plt.plot(fcn(y, f, lam[i]), y, c = 'black')
    plt.plot(-fcn(y, f, lam[i]), y, c = 'black')

plt.show()