
from tensorflow.compat.v1.train import get_or_create_global_step
import pickle 
import numpy as np
import matplotlib.pyplot as plt 

'''with open("circle_scatter.pickle", "rb") as f:
    data = pickle.load(f)


noise = [np.random.uniform(0, 1, size = 3) for i in range(12)]
noise_y = [[0] for _ in range(12)]

data = np.array(data)

real_data = list(data[np.random.choice(len(data), 12), :])
real_y = [[1] for _ in range(12)]

_y = noise_y + real_y
_x = noise + real_data


batch_data = np.array(list(map(list, zip(*[_x, _y]))))
'''

print(type([[1], [2]]))