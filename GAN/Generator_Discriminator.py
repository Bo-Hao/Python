import tensorflow as tf 
import numpy as np 
import copy 
import decorator
from NoisyDense import NoisyDense



if __name__ == "__main__":
    G = Generator(4, 4)
    G.build()
    inputs = [[[1, 1, 1, 1], [2, 2, 2, 2]]]
    ans = G.model.predict(inputs)
    print(ans)