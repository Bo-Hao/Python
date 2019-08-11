import numpy as np 
import pickle
import cv2
import matplotlib.pyplot as plt
import itertools
import heapq


class Epipolar:
    def __init__(self, points_1, points_2):
        self.pt1 = [[points_1[i][0], points_1[i][1], 1]for i in range(len(points_1))]
        self.pt2 = [[points_2[i][0], points_2[i][1], 1]for i in range(len(points_2))]
        self.ransac = 500
        self.distance = 3
        #log(1-(1-p))/log(1-p**8)
        
    def epipolar(self, method = '8points'):
        if len(self.pt1) != len(self.pt2):
            print("Can't make pairs ")
        if method == '8points':
            A = [[self.pt1[k][i]*self.pt2[k][j] for j in range(3) for i in range(3)]for k in range(len(self.pt1))]
            A = np.array(A)
            U, D, Vt = np.linalg.svd(A)
            u, d, vt = np.linalg.svd(np.reshape(Vt[-1], (3, 3)))
            d = np.diag([d[0], d[1], 0])
            F = np.dot(np.dot(u, d), vt)
            return F
        
        elif method == 'RANSAC':
            def point2epiline(point, line):
                x, y = point[0], point[1]
                a, b, c = line[0], line[1], line[2]
                return abs(a*x + b*y + c)/(a**2 + b**2)**0.5, [a, b, c]
            
            choice = list(itertools.combinations(np.arange(len(self.pt1)), 10))
            ran = np.random.randint(len(choice), size = self.ransac)
            record = []
            #inlier = [0 for i in range(len(self.pt1))]
            for c_ in ran:
                A = [[self.pt1[k][i]*self.pt2[k][j] for j in range(3) for i in range(3)]for k in choice[c_]]
                A = np.array(A)
                U, D, Vt = np.linalg.svd(A)
                u, d, vt = np.linalg.svd(np.reshape(Vt[-1], (3, 3)))
                d = np.diag([d[0], d[1], 0])
                F = np.dot(np.dot(u, d), vt)
                p1 = np.array(self.pt1) 
                p2 = np.array(self.pt2)
                
                inlier = [0 for i in range(len(p1))]
                prev_line = 0
                punish = 0
                total_dist = 0
                for i_ in range(len(p1)):
                    distance_1,line_1 = point2epiline(p1[i_], np.dot(F.T, p2[i_]))
                    distance_2,line_2 = point2epiline(p2[i_], np.dot(F, p1[i_]))
                    
                    if abs(distance_1 - distance_2) < self.distance:

                        if prev_line != 0:
                            if abs(prev_line[0]/line_1[0] - prev_line[1]/line_1[1]) <= 0.002:
                                prev_line = line_1
                                print("warning!")
                                punish = 1
                            else:
                                inlier[i_] += 1
                                total_dist += distance_1
                        else:
                            prev_line = line_1
                            inlier[i_] += 1
                            total_dist += distance_1
                if punish == 0:
                    in_index = [k for k in range(len(inlier)) if inlier[k] == 1]
                    print(total_dist)
                    record.append([sum(inlier), F, in_index, total_dist])
            record = sorted(record, key = lambda x:x[3])
            F = record[0][1]
                     
                
        
            
            
            
            return F
                

    
    
def decomposition(F):
    Z = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    U, D, Vt = np.linalg.svd(F)
    S = np.dot(np.dot(U, Z), U.T)
    R1 = np.dot(np.dot(U, W), Vt)
    R2 = np.dot(np.dot(U, W.T), Vt)
    S = [S[1, 2], -S[0, 2], S[0, 1]]
    return S, R1, R2
        
def imageinfo(pixel, focus_length, real_size):
    u0 = pixel[0] / 2
    v0 = pixel[1] / 2
    dx = real_size[0]*1000 / pixel[0]
    dy = real_size[1]*1000 / pixel[1]
    fx = focus_length / dx
    fy = focus_length / dy
    intrinsic_matrix = np.array([[fx, 0, u0], [0, fy, v0], [0, 0, 1]])
    # extrinsic_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0 ,0 ,1, 0]])
    # camera_matrix = np.dot(intrinsic_matrix, extrinsic_matrix)
    return intrinsic_matrix

def drawepiline(F, pt1, pt2, im1):        
    epline = [np.dot(F.T, pt2[i])for i in range(len(pt2))]
    shape = np.array(im1).shape
    pa, pb = [], []
    
    for i in epline:
        xx = -i[2]/i[0]
        yy = -i[2]/i[1]
        if xx > 0:
            pa.append([xx, 0])
        else:
            pa.append([(-(i[2]/i[0] + i[1] * shape[0]/i[0])), shape[0]])
            
        if yy > 0:
            pb.append([0, yy])
        else:
            pb.append([shape[1], (-(i[2]/i[1] + i[0] * shape[1]/i[1]))])
            

    plt.imshow(im1)
    for i in range(len(pa)):
        plt.plot([pa[i][0], pb[i][0]], [pa[i][1], pb[i][1]])
    pa = np.array(pa)
    pb = np.array(pb)
    plt.scatter(pa[:, 0], pa[:, 1])
    plt.scatter(pb[:, 0], pb[:, 1])
    plt.scatter(pt1.T[0], pt1.T[1], c = 'red', s = 1)
    plt.ylim(shape[0], 0)
    plt.xlim(0, shape[1])
    plt.show()




def main():
    # image information
    bad = [9, 14, 22, 27]
    bad = []
    pixel = (3648, 5472)
    focus_length = 8.8 #mm
    real_size = (13.2, 8.8)        
    I = imageinfo(pixel, focus_length, real_size)
    image_address1 = '/Users/pengbohao/Downloads/2019summer/IMG_2881.JPG'
    image_address2 = '/Users/pengbohao/Downloads/2019summer/IMG_2882.JPG'    
    im1 = plt.imread(image_address1)
    im2 = plt.imread(image_address2)
    
        
    with open('/Users/pengbohao/Downloads/2019summer/save.pickle', 'rb') as f:
        points = pickle.load(f)
    p1, p2 = np.array(points[0]), np.array(points[1])
    p1 = np.array([[p1[i][0], p1[i][1], 1]for i in range(len(p1)) if i not in bad])
    p2 = np.array([[p2[i][0], p2[i][1], 1]for i in range(len(p2)) if i not in bad])
    
    E1 = Epipolar(p1, p2)
    #E2 = Epipolar(pt1, pt2)
    
    F1 = E1.epipolar(method = 'RANSAC')
    #F2 = E2.epipolar()
    
    Fcv2_1, mask = cv2.findFundamentalMat(p1, p2, method = cv2.RANSAC)
    #Fcv2_2 = cv2.findFundamentalMat(pt1, pt2, method = cv2.FM_8POINT)
    
    for i_ in range(len(p1)):
        ans = np.dot(np.dot(p2[i_].T, F1), p1[i_])

        print(mask[i_], ans)

    
    #Es1 = np.dot(np.dot(I, F1), I.T)
    #Escv2_1 = np.dot(np.dot(I, Fcv2_1[0]), I)    
    drawepiline(F1.T, p2, p1, im2)
    drawepiline(F1, p1, p2, im1)

    
record = main()  