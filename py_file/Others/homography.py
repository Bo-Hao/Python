import numpy as np 
from math import *
import pickle




class Computer_vision:
    def __init__(self, control_points_A, control_points_B):
        self.ptsA = control_points_A
        self.ptsB = control_points_B
        
        
    def camera_matrix(self, ):
        pass

    





#if __name__ == "__main__":
with open('/Users/pengbohao/Downloads/2019summer/save.pickle', 'rb') as f:
    points = pickle.load(f)
p1, p2 = points[0], points[1]
C = Computer_vision(pt1, pt2)